"""【概要】カメラ画像から「赤い物体がどこにあるか」を求める(視覚サーボの"目"の部分)。

【検出器の約束(差し替えの単位)】 この関数の形を守れば、中身は何でも差し替えられる:
    detector(img) -> (u, v) または None
      img : 高さ × 幅 × 3 (RGB, uint8) の画像。  (u, v) : 物体の中心の画素位置。見つからなければ None。
制御側(visual_servo.py)は、この形だけを知っていて、色で探すのか、物体検出モデルで探すのかを知らない。
既製の物体検出モデル(YOLO など)は「枠 (x0, y0, x1, y1)」を返すことが多いので、bbox_to_center で
(u, v) に直して包めばよい。差し替え方の例は scripts/check_detectors.py。

流れ:  カメラ画像 --find_red_centroid--> 画像上の位置 (u, v) [画素]
                  --pixel_to_world-->   世界座標の位置 (x, y) [m]
この位置を、制御側(IK など)の目標として渡すのが第2週の目標。
(計画書は「既製の物体検出モデル」を想定しているが、まずは色で見つける簡単な方法から始める。
 同じ「画像 → 位置」の形に保っておけば、後で検出モデルに差し替えても制御側は変えずに済む。)
"""
import numpy as np
from scipy import ndimage  # 画像処理の道具箱。ここでは「つながった塊に番号を振る」ために使う。[頻度] 画像処理で定番


def find_red_centroid(img):
    """画像 img から赤い画素を探し、その重心 (u, v) [画素] を返す。無ければ None を返す。

    img : 高さ × 幅 × 3 (RGB) の配列、dtype=uint8 (0〜255)。mujoco.Renderer.render() の戻り値。
    u   : 横方向の位置(左端が0、右へ増える)。v : 縦方向の位置(上端が0、下へ増える)。

    ヒント: 「赤い」= R が大きく、G と B がそれより十分小さい画素。例: R>100 かつ G<0.4*R かつ B<0.4*R。
            (アームの黄色は G も大きいので、この条件で自然に除外される)
            numpy の「条件で True/False の配列(マスク)を作る」「np.nonzero や np.argwhere で
            True の位置を取り出す」「.mean() で平均」を組み合わせると短く書ける。
            赤い画素が1つも無いとき、平均を取ると nan になるので、先に個数を確認すること。
    """
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    mask = (r > 100) & (g < 0.4*r) & (b < 0.4*r)
    if not mask.any():
      return None
    y, x = np.nonzero(mask)
    u = (np.max(x) + np.min(x)) / 2
    v = (np.max(y) + np.min(y)) / 2
    return (u, v)

def pixel_to_world(u, v, cam_pos, fovy_deg, width, height, z_obj):
    """真下を向いた固定カメラの画像上の位置 (u, v) [画素] を、世界座標の (x, y) [m] に直す。

    cam_pos  : カメラの世界座標 (x, y, z) [m]。
    fovy_deg : カメラの縦方向の視野角 [度]。
    width, height : 画像の幅・高さ [画素]。
    z_obj    : 物体の(見ている面の)高さ [m]。カメラからこの面までの距離が「奥行き」になる。

    前提(models/so101/scene_vs.xml の overhead カメラ): 画像の右 = 世界の +x、画像の上 = 世界の +y。
    ヒント(ピンホールカメラの考え方):
      1. 焦点距離 f [画素] = (height/2) / tan(fovy/2)   (fovy はラジアンに直す: np.radians)
      2. 奥行き depth = cam_pos の z - z_obj [m]
      3. 画像の中心 (width/2, height/2) からのずれ (u-width/2, v-height/2) [画素] を、
         「 / f * depth」で実寸 [m] に直す。
      4. 画像の縦方向は v が下へ増えるが、世界の y は上へ増えるので、符号に注意。
    """
    f = (height/2) / np.tan(np.radians(fovy_deg)/2)
    depth = cam_pos[2] - z_obj
    x = (u - width/2) / f * depth + cam_pos[0]
    # v は下へ増える: 世界の y は上へ増える → 逆向きなので引く。
    y = -(v - height/2) / f * depth + cam_pos[1]
    return (x, y)


def find_blue_centroid(img):
    """画像 img から青い画素を探し、その重心 (u, v) [画素] を返す。無ければ None を返す。

    find_red_centroid の青版。視覚サーボ(closed-loop)で、手先に付けた青い目印を見つけるのに使う。
    R, B の役割を入れ替えただけ(B が大きく、R と G がそれより十分小さい画素を選ぶ)。
    アームの黄色・黒、赤い立方体は、この条件では選ばれない。
    """
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    mask = (b > 100) & (r < 0.4 * b) & (g < 0.4 * b)
    if not mask.any():
        return None
    v_idx, u_idx = np.nonzero(mask)  # 行(縦 v)の番号と列(横 u)の番号。img[v, u] の順に注意
    return ((u_idx.max() + u_idx.min()) / 2, (v_idx.max() + v_idx.min()) / 2)


# ---- 差し替え用の検出器の例 ----

def _largest_blob_centroid(mask):
    """True/False の配列 mask の中で、一番大きな「つながった塊」の重心 (u, v) を返す。無ければ None。

    scipy.ndimage.label: つながった True の塊ごとに 1, 2, 3 … と番号を振る(背景は 0)。[頻度] 塊を数えるときの定番
    find_*_centroid は「赤い画素すべて」の範囲の中心を取るので、遠くに赤い点が1つ混ざるだけで、中心が大きくずれる。
    こちらは一番大きい塊だけを使うので、小さなごみ(ノイズ、別の小さな物体)に強い。
    """
    labels, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(mask, labels, index=range(1, n + 1))  # 塊ごとの画素数
    best = int(np.argmax(sizes)) + 1  # 一番大きい塊の番号(argmax は 0 始まりなので +1)
    v_idx, u_idx = np.nonzero(labels == best)
    return ((u_idx.max() + u_idx.min()) / 2, (v_idx.max() + v_idx.min()) / 2)


def find_red_blob_centroid(img):
    """赤い画素のうち、一番大きな塊の中心 (u, v) を返す。find_red_centroid の「ごみに強い」版。"""
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return _largest_blob_centroid((r > 100) & (g < 0.4 * r) & (b < 0.4 * r))


def find_blue_blob_centroid(img):
    """青い画素のうち、一番大きな塊の中心 (u, v) を返す。find_blue_centroid の「ごみに強い」版。"""
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return _largest_blob_centroid((b > 100) & (r < 0.4 * b) & (g < 0.4 * b))


def bbox_to_center(bbox):
    """物体検出モデルが返す枠 (x0, y0, x1, y1) [画素] を、中心 (u, v) に直す。枠が None なら None を返す。

    YOLO など既製のモデルの出力は、この形の枠であることが多い。モデルの呼び出しを、
    「画像 → 枠」の関数として書き、bbox_to_center で包めば、検出器の約束 detector(img) -> (u, v) を満たす。
    """
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2, (y0 + y1) / 2)


# ---- 複数の色の立方体を見分けて、位置と向きを出す(部品仕分け用。新計画の工程3)----

def color_mask(img, color):
    """画像 img の中で、color ("red"、"green"、"blue"、手の目印の "magenta")の画素が True の配列(マスク)を返す。
    条件は「その色のチャンネルが大きく(100 超)、他の 2 つがその 0.4 倍未満」。置き場の薄い色(他のチャンネルも大きい)や、
    アームの黄色・黒は、この条件で除外される。"""
    if color == "magenta":  # 手の目印(赤と青が大きく、緑が小さい)。立方体の 3 色の条件には当てはまらない
        r, g, b = (img[:, :, c].astype(float) for c in range(3))
        return (r > 100) & (b > 100) & (g < 0.4 * np.minimum(r, b))
    channel = {"red": 0, "green": 1, "blue": 2}[color]
    main = img[:, :, channel].astype(float)  # astype(float): 整数(uint8)のまま 0.4 倍などを比べると、丸め誤差が出るので小数にする
    others = [img[:, :, c].astype(float) for c in range(3) if c != channel]
    return (main > 100) & (others[0] < 0.4 * main) & (others[1] < 0.4 * main)


def _largest_component(mask):
    """マスクの中で、一番大きな「つながった塊」だけを残したマスクを返す。無ければ None。(_largest_blob_centroid と同じ考え方)"""
    labels, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(mask, labels, index=range(1, n + 1))
    return labels == (int(np.argmax(sizes)) + 1)


def estimate_square_yaw(mask):
    """マスク(正方形に近い塊)の向き [度、0〜90 未満] を返す。世界座標の x 軸(画像の右)から、反時計回りに測る。

    考え方: 塊の画素を、角度 θ だけ回した座標軸に投影し、軸にそろった長方形の面積が最小になる θ を探す。
    正方形は、辺が軸にそろったとき、外接する長方形の面積が最小になる。0〜90° を 1° ずつ試す(90° ごとに同じ形なので、これで足りる)。
    (画像の v は下へ増えるが、世界の y は上へ増える。向きを世界座標で出すため、v の符号を反転して扱う。)
    """
    v_idx, u_idx = np.nonzero(mask)
    pts = np.stack([u_idx, -v_idx], axis=1).astype(float)  # (画素数, 2): 世界座標と同じ向き(右 = +x、上 = +y)の座標
    best_theta, best_area = 0, np.inf
    for theta in range(90):
        t = np.radians(theta)
        a = pts @ np.array([np.cos(t), np.sin(t)])  # θ 方向の軸への投影(各画素の位置)
        b = pts @ np.array([-np.sin(t), np.cos(t)])  # それに垂直な軸への投影
        area = (a.max() - a.min()) * (b.max() - b.min())
        if area < best_area:
            best_theta, best_area = theta, area
    return float(best_theta)


def find_cube_pose(img, color):
    """画像 img から、color の立方体(一番大きい塊)の、中心 (u, v) [画素] と向き yaw [度] を返す。無ければ None。
    戻り値: (u, v, yaw)。中心は、塊の外接する四角形の中心(find_*_centroid と同じ)。"""
    mask = _largest_component(color_mask(img, color))
    if mask is None or mask.sum() < 20:  # 20 画素未満は、ノイズとして見つけなかったことにする
        return None
    v_idx, u_idx = np.nonzero(mask)
    return ((u_idx.max() + u_idx.min()) / 2, (v_idx.max() + v_idx.min()) / 2, estimate_square_yaw(mask))
