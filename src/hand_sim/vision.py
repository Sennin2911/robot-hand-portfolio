"""【概要】カメラ画像から「赤い物体がどこにあるか」を求める(視覚サーボの"目"の部分)。

流れ:  カメラ画像 --find_red_centroid--> 画像上の位置 (u, v) [画素]
                  --pixel_to_world-->   世界座標の位置 (x, y) [m]
この位置を、制御側(IK など)の目標として渡すのが第2週の目標。
(計画書は「既製の物体検出モデル」を想定しているが、まずは色で見つける簡単な方法から始める。
 同じ「画像 → 位置」の形に保っておけば、後で検出モデルに差し替えても制御側は変えずに済む。)
"""
import numpy as np


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
    raise NotImplementedError("find_red_centroid がまだ書かれていません")


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
    raise NotImplementedError("pixel_to_world がまだ書かれていません")
