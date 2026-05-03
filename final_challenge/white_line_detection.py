import cv2 as cv
import numpy as np

#################### X-Y CONVENTIONS #########################
# 0,0  X  > > > > >
#
#  Y
#
#  v  This is the image. Y increases downwards, X increases rightwards
#  v  Lane line points are returned as lists of (x, y) tuples
#  v  in original image coordinates.
###############################################################


# ── Tunable parameters ────────────────────────────────────────────────────────
ROI_TOP_FRAC    = 0.40   # fraction of image height to crop from top
HSV_LOW         = np.array([  0,   0, 160])  # any hue, low sat, bright
HSV_HIGH        = np.array([179,  60, 255])
MIN_AREA        = 300    # px² – discard tiny speckles. Was 500 (matched
                          # calibration GUI defaults). Lowered to 300 because
                          # the Johnson Track bag's left lane line is often
                          # foreshortened to 350-450 px² on curves and was
                          # being rejected at 500. 300 recovers ~3% more
                          # left-line detections per frame without adding
                          # meaningful noise — keep the GUI in sync if you
                          # change it again.
MIN_LONG_SIDE   = 40     # px – minimum length of the blob's long axis
MIN_ELONGATION  = 3.0    # long_side / short_side – lane lines are thin & long
MAX_DXDY        = 5.5    # |dx/dy| threshold for bottom-tangent VP filter
# Hough-transform parameters (HoughLinesP)
HOUGH_THRESHOLD    = 20   # min accumulator votes for a line
HOUGH_MIN_LINE_LEN = 25   # px – discard very short segments
HOUGH_MAX_LINE_GAP = 15   # px – bridge small gaps within a line
HOUGH_VERT_THRESH  = 0.3  # |dy/dx| lower bound — rejects near-horizontal cross-track lines
HOUGH_VERT_MAX     = 5.0  # |dy/dx| upper bound — keep near-vertical lane stripes when camera is close
# Standard HoughLines + clustering parameters
HOUGH_STD_THRESHOLD  = 30   # min accumulator votes for standard HoughLines
HOUGH_CLUSTER_D_RHO   = 60  # px  – max ρ spread to merge into one cluster
HOUGH_CLUSTER_D_THETA = 0.20 # rad – max θ spread to merge (~11.5°)
# ──────────────────────────────────────────────────────────────────────────────


def image_print(img, title="image"):
    """
    Display an image in a named window. Press any key to continue.
    """
    cv.imshow(title, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def _hsv_white_mask(roi):
    """Return a binary mask of white pixels in the ROI (BGR input)."""
    hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    return cv.inRange(hsv, HSV_LOW, HSV_HIGH)


def _morphological_cleanup(mask):
    """Remove speckle and close small gaps in the mask."""
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN,  kernel, iterations=1)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)
    return mask


def _extract_spine(label_mask):
    """
    For each row that contains white pixels in label_mask, record the
    mean x of those pixels. Returns a list of (x, y) in ROI coordinates,
    ordered top-to-bottom (increasing y).
    """
    rows, cols = np.where(label_mask)
    if rows.size == 0:
        return []
    h = label_mask.shape[0]
    counts = np.bincount(rows, minlength=h)
    sums   = np.bincount(rows, weights=cols.astype(np.float64), minlength=h)
    ys = np.nonzero(counts)[0]
    # int() on a positive float truncates toward zero — match the original.
    xs = (sums[ys] / counts[ys]).astype(np.int64)
    return list(zip(xs.tolist(), ys.tolist()))


def _local_dxdy(pts):
    """Fit a line to pts and return |dx/dy|, or inf if degenerate."""
    ys = np.array([p[1] for p in pts], dtype=float)
    xs = np.array([p[0] for p in pts], dtype=float)
    if np.ptp(ys) < 2:
        return float('inf')
    return abs(np.polyfit(ys, xs, deg=1)[0])


def _tangent_ok(spine_points):
    """
    Accept the blob if EITHER end of its spine is steep (|dx/dy| < MAX_DXDY).
    - Straight sections: steep at bottom (close to car), looser at top.
    - Mid-turn: steep at top (near VP), horizontal at bottom.
    - Horizontal noise (ceiling streaks, cross-lines): flat at both ends → rejected.
    """
    if len(spine_points) < 4:
        return False
    n = max(2, len(spine_points) // 5)
    top_dxdy = _local_dxdy(spine_points[:n])
    bot_dxdy = _local_dxdy(spine_points[-n:])
    return top_dxdy < MAX_DXDY or bot_dxdy < MAX_DXDY


def detect_white_lines(img, debug=False):
    """
    Detect white lane lines (straight or curved) in a racetrack image.

    Parameters
    ----------
    img   : np.ndarray  BGR image from the car camera
    debug : bool        If True, display intermediate masks interactively
                        (press any key to advance through each stage)

    Returns
    -------
    lane_lines : list of list of (x, y) tuples
        One entry per detected lane line, each a list of (x, y) points
        in original image coordinates, ordered bottom-to-top (decreasing y).
    """
    h, w = img.shape[:2]
    roi_top = int(ROI_TOP_FRAC * h)
    roi = img[roi_top:, :]

    # ── Stage 1: ROI crop ────────────────────────────────────────────────────
    # if debug:
    #     image_print(roi, "1_roi_crop")

    # ── Stage 2: HSV white mask ──────────────────────────────────────────────
    mask_raw = _hsv_white_mask(roi)
    # if debug:
    #     image_print(mask_raw, "2_hsv_mask_raw")

    # ── Stage 3: Morphological cleanup ───────────────────────────────────────
    mask_clean = _morphological_cleanup(mask_raw)
    # if debug:
    #     image_print(mask_clean, "3_mask_after_morphology")

    # ── Stage 4: Connected components ────────────────────────────────────────
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(mask_clean)

    # if debug:
    #     color_map = np.zeros((*mask_clean.shape, 3), dtype=np.uint8)
    #     rng = np.random.default_rng(42)
    #     for i in range(1, num_labels):
    #         color = rng.integers(80, 255, 3).tolist()
    #         color_map[labels == i] = color
    #     image_print(color_map, "4_all_components")

    # ── Stage 5: Shape filter ────────────────────────────────────────────────
    # Use minAreaRect so diagonal lane lines are not incorrectly rejected.
    # A lane line at 45° has an axis-aligned bounding box with width ≈ height,
    # so height > width would wrongly discard it. minAreaRect measures the
    # actual long/short axes of the blob regardless of angle.
    candidates = []
    for i in range(1, num_labels):
        area = stats[i, cv.CC_STAT_AREA]
        if area < MIN_AREA:
            continue
        component_mask = np.uint8(labels == i)
        contours, _ = cv.findContours(component_mask, cv.RETR_EXTERNAL,
                                      cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        pts = np.vstack([c.reshape(-1, 2) for c in contours])
        if len(pts) < 5:
            continue
        _, (rw, rh), _ = cv.minAreaRect(pts)
        long_side  = max(rw, rh)
        short_side = min(rw, rh) + 1e-6
        if long_side >= MIN_LONG_SIDE and long_side / short_side >= MIN_ELONGATION:
            candidates.append(i)

    # if debug:
    #     filtered_map = np.zeros((*mask_clean.shape, 3), dtype=np.uint8)
    #     rng2 = np.random.default_rng(42)
    #     all_colors = {
    #         i: rng2.integers(80, 255, 3).tolist()
    #         for i in range(1, num_labels)
    #     }
    #     for i in candidates:
    #         filtered_map[labels == i] = all_colors[i]
    #     image_print(filtered_map, "5_filtered_components")

    # ── Spine extraction + bottom-tangent VP filter ───────────────────────────
    lane_lines = []
    for i in candidates:
        label_mask = (labels == i)
        spine = _extract_spine(label_mask)   # (x, y) in ROI coords
        if not _tangent_ok(spine):
            continue
        # Convert back to original image coordinates
        spine_orig = [(x, y + roi_top) for (x, y) in spine]
        # Return ordered bottom-to-top (highest y first)
        spine_orig.sort(key=lambda p: p[1], reverse=True)
        lane_lines.append(spine_orig)

    # ── Stage 6: Final result on original image ───────────────────────────────
    # if debug:
    #     result_img = img.copy()
    #     colors = [
    #         (0, 255, 0), (0, 255, 255), (255, 165, 0),
    #         (255, 0, 255), (0, 128, 255)
    #     ]
    #     for idx, spine in enumerate(lane_lines):
    #         color = colors[idx % len(colors)]
    #         for (x, y) in spine:
    #             cv.circle(result_img, (x, y), 2, color, -1)
    #         if spine:
    #             lx, ly = spine[0]
    #             cv.putText(result_img, f"L{idx}", (lx + 4, ly),
    #                        cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    #     cv.line(result_img, (0, roi_top), (w, roi_top), (128, 128, 128), 1)
    #     image_print(result_img, "6_final_detections")

    return lane_lines


def fit_lane_polynomials(img, debug=False):
    """
    Fit a 2nd-degree polynomial to each detected white lane line.

    Parameters
    ----------
    img   : np.ndarray  BGR image
    debug : bool        Passed through to detect_white_lines for stage 1-5;
                        additionally shows polynomial curves on the image.

    Returns
    -------
    polynomials : list of np.ndarray
        Each entry is [a, b, c] such that  x = a*y² + b*y + c
        for a lane line. Evaluate at any y in image coords to get x.
    """
    lane_lines = detect_white_lines(img, debug=False)

    polynomials = []
    for spine in lane_lines:
        if len(spine) < 5:
            continue
        ys = np.array([p[1] for p in spine], dtype=float)
        xs = np.array([p[0] for p in spine], dtype=float)
        try:
            coeffs = np.polyfit(ys, xs, deg=2)
            polynomials.append(coeffs)
        except np.linalg.LinAlgError:
            continue

    if debug:
        h, w = img.shape[:2]
        result_img = img.copy()
        colors = [
            (0, 255, 0), (0, 255, 255), (255, 165, 0),
            (255, 0, 255), (0, 128, 255)
        ]
        for idx, coeffs in enumerate(polynomials):
            color = colors[idx % len(colors)]
            poly = np.poly1d(coeffs)
            ys_draw = np.arange(int(0.40 * h), h, 2)
            for y in ys_draw:
                x = int(poly(y))
                if 0 <= x < w:
                    cv.circle(result_img, (x, y), 2, color, -1)
            # Label
            y_label = int(0.85 * h)
            x_label = int(np.clip(poly(y_label), 0, w - 1))
            cv.putText(result_img, f"P{idx}", (x_label + 4, y_label),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        image_print(result_img, "poly_final_detections")

    return polynomials


def detect_vertical_lines_hough(img, debug=False):
    """
    Detect vertical lane lines using the probabilistic Hough transform.

    "Vertical" means the longitudinal lane lines running toward the vanishing
    point. Due to perspective they can appear anywhere from ~17° to 80° from
    horizontal, so the filter keeps segments with |dy/dx| > HOUGH_VERT_THRESH
    (default 0.3) while rejecting truly horizontal ceiling/floor artifacts.

    Parameters
    ----------
    img   : np.ndarray  BGR image from the car camera
    debug : bool        If True, display detected segments overlaid on the image

    Returns
    -------
    segments : list of ((x1, y1), (x2, y2))
        Each entry is a pair of endpoints in original image coordinates.
    """
    h, w = img.shape[:2]
    roi_top = int(ROI_TOP_FRAC * h)
    roi = img[roi_top:, :]

    # White mask + cleanup (reuse existing helpers)
    mask = _hsv_white_mask(roi)
    mask = _morphological_cleanup(mask)

    # Canny on the binary mask gives crisp edge pixels for the accumulator
    edges = cv.Canny(mask, 50, 150)

    raw = cv.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LEN,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )

    if raw is None:
        return []

    segments = []
    for seg in raw:
        x1, y1, x2, y2 = seg[0]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dy / (dx + 1e-6) >= HOUGH_VERT_THRESH:
            segments.append(((x1, y1 + roi_top), (x2, y2 + roi_top)))

    if debug:
        result_img = img.copy()
        for (p1, p2) in segments:
            cv.line(result_img, p1, p2, (0, 255, 0), 2)
        cv.line(result_img, (0, roi_top), (w, roi_top), (128, 128, 128), 1)
        image_print(result_img, "hough_vertical_lines")

    return segments


def _cluster_hough_lines(lines, d_rho=60, d_theta=0.15):
    """
    Greedy merge of (ρ, θ) accumulator peaks into one centroid per physical line.

    Sort by θ so that angle-adjacent peaks are processed together; separate by ρ
    to distinguish parallel lines (d_rho must be much smaller than the inter-line
    ρ gap, which is typically 100-200 px for our track geometry).
    """
    sorted_lines = sorted(lines, key=lambda l: l[1])
    clusters = []
    for rho, theta in sorted_lines:
        for c in clusters:
            c_rho   = float(np.mean([p[0] for p in c]))
            c_theta = float(np.mean([p[1] for p in c]))
            if abs(theta - c_theta) < d_theta and abs(rho - c_rho) < d_rho:
                c.append((rho, theta))
                break
        else:
            clusters.append([(rho, theta)])
    return [
        (float(np.mean([p[0] for p in c])), float(np.mean([p[1] for p in c])))
        for c in clusters
    ]


def detect_lane_lines_hough(img, debug=False):
    """
    Detect lane lines via standard HoughLines with (ρ, θ) accumulator clustering.

    One cluster of nearby peaks → one physical lane line, free of the segment
    fragmentation that affects HoughLinesP.  Each line is returned as the
    slope-intercept form  x = m·y + b  (original image coords, y increases
    downward), so evaluating at any y gives the expected x for that line —
    directly usable as a pure-pursuit lateral target.

    Parameters
    ----------
    img   : np.ndarray  BGR image from the car camera
    debug : bool        If True, draw detected lines on the image

    Returns
    -------
    lane_lines : list of dict, each containing:
        'coeffs'  : (m, b) — floats, x = m*y + b in original image coords
        'segment' : ((x1, y1), (x2, y2)) — line clipped to the ROI y-range
        'votes'   : int — number of accumulator peaks in the cluster (confidence)
    """
    h, w = img.shape[:2]
    roi_top = int(ROI_TOP_FRAC * h)
    roi = img[roi_top:, :]

    mask = _morphological_cleanup(_hsv_white_mask(roi))
    edges = cv.Canny(mask, 50, 150)

    raw = cv.HoughLines(edges, rho=1, theta=np.pi / 180,
                        threshold=HOUGH_STD_THRESHOLD)
    if raw is None:
        return []

    # Two-sided angle gate applied to raw peaks *before* clustering so that
    # cross-track markers and near-image-vertical lines never reach the merger.
    # Line: x·cos(θ) + y·sin(θ) = ρ  →  |dy/dx| = |cos(θ)/sin(θ)|
    # Keep peaks where HOUGH_VERT_THRESH <= |dy/dx| <= HOUGH_VERT_MAX.
    filtered = []
    for r in raw:
        rho, theta = float(r[0][0]), float(r[0][1])
        dy_dx = abs(np.cos(theta)) / (abs(np.sin(theta)) + 1e-6)
        if HOUGH_VERT_THRESH <= dy_dx <= HOUGH_VERT_MAX:
            filtered.append((rho, theta))
    if not filtered:
        return []

    clustered = _cluster_hough_lines(filtered,
                                     d_rho=HOUGH_CLUSTER_D_RHO,
                                     d_theta=HOUGH_CLUSTER_D_THETA)

    # Count votes per cluster for confidence reporting
    votes_map = {}
    for rho, theta in filtered:
        for idx, (cr, ct) in enumerate(clustered):
            if abs(theta - ct) < HOUGH_CLUSTER_D_THETA and abs(rho - cr) < HOUGH_CLUSTER_D_RHO:
                votes_map[idx] = votes_map.get(idx, 0) + 1
                break

    lane_lines = []
    for idx, (rho, theta) in enumerate(clustered):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        if abs(cos_t) < 1e-6:
            continue

        # Line in ROI coords: x = m·y_roi + b_roi
        m     = -sin_t / cos_t
        b_roi = rho / cos_t

        # Shift to original image coords: y_roi = y - roi_top
        b = b_roi - m * roi_top

        # Clip segment endpoints to the ROI y-range and image width
        y_top = roi_top
        y_bot = h - 1
        x_top = int(np.clip(m * y_top + b, 0, w - 1))
        x_bot = int(np.clip(m * y_bot + b, 0, w - 1))

        lane_lines.append({
            'coeffs':  (m, b),
            'segment': ((x_top, y_top), (x_bot, y_bot)),
            'votes':   votes_map.get(idx, 0),
        })

    if debug:
        result_img = img.copy()
        colors = [(0, 255, 0), (0, 255, 255), (255, 165, 0),
                  (255, 0, 255), (0, 128, 255)]
        cv.line(result_img, (0, roi_top), (w, roi_top), (128, 128, 128), 1)
        for i, ll in enumerate(lane_lines):
            color = colors[i % len(colors)]
            p1, p2 = ll['segment']
            cv.line(result_img, p1, p2, color, 2)
            cv.circle(result_img, p1, 4, color, -1)
            cv.circle(result_img, p2, 4, color, -1)
            m, b = ll['coeffs']
            y_label = roi_top + (h - roi_top) // 2
            x_label = int(np.clip(m * y_label + b, 4, w - 40))
            cv.putText(result_img, "m=%.2f" % m, (x_label, y_label - 4),
                       cv.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        image_print(result_img, "lane_lines_hough")

    return lane_lines


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import os

    # Default: run through all images in a lane directory
    test_dirs = [
        "testing_images/racetrack_images/lane_1",
        "testing_images/racetrack_images/lane_3",
        "testing_images/racetrack_images/lane_6",
    ]

    if len(sys.argv) > 1:
        # Allow passing a single image path or directory as argument
        paths = sys.argv[1:]
    else:
        paths = []
        for d in test_dirs:
            if os.path.isdir(d):
                for fname in sorted(os.listdir(d)):
                    if fname.lower().endswith(".png"):
                        paths.append(os.path.join(d, fname))

    colors = [(0,255,0),(0,255,255),(255,165,0),(255,0,255),(0,128,255)]

    for path in paths:
        print(f"\n--- {path} ---")
        frame = cv.imread(path)
        if frame is None:
            print(f"  Could not load {path}")
            continue
        h, w = frame.shape[:2]

        # ── Blob-based detector ───────────────────────────────────────────────
        lines = detect_white_lines(frame, debug=False)
        print(f"  [blob ] {len(lines)} lane line(s)")
        for i, spine in enumerate(lines):
            print(f"    Line {i}: {len(spine)} points, "
                  f"y range [{spine[-1][1]}–{spine[0][1]}]")

        result_blob = frame.copy()
        for idx, spine in enumerate(lines):
            color = colors[idx % len(colors)]
            for (x, y) in spine:
                cv.circle(result_blob, (x, y), 2, color, -1)
            if spine:
                lx, ly = spine[0]
                cv.putText(result_blob, f"L{idx}", (lx + 4, ly),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv.line(result_blob, (0, int(ROI_TOP_FRAC * h)), (w, int(ROI_TOP_FRAC * h)),
                (128, 128, 128), 1)
        cv.putText(result_blob, path + "  [blob]", (4, 14),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        image_print(result_blob, "blob_detections")

        # ── Hough-based detector ──────────────────────────────────────────────
        segs = detect_vertical_lines_hough(frame, debug=False)
        print(f"  [hough] {len(segs)} vertical segment(s)")
        for i, (p1, p2) in enumerate(segs):
            print(f"    Seg {i}: {p1} → {p2}")

        result_hough = frame.copy()
        for idx, (p1, p2) in enumerate(segs):
            color = colors[idx % len(colors)]
            cv.line(result_hough, p1, p2, color, 2)
            cv.circle(result_hough, p1, 4, color, -1)
            cv.circle(result_hough, p2, 4, color, -1)
        cv.line(result_hough, (0, int(ROI_TOP_FRAC * h)), (w, int(ROI_TOP_FRAC * h)),
                (128, 128, 128), 1)
        cv.putText(result_hough, path + "  [hough]", (4, 14),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        image_print(result_hough, "hough_detections")
