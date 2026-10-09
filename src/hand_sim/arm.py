"""【概要】2リンク平面アームの運動学(順運動学 fk、逆運動学 ik)。

対象モデルは models/arm2link.xml。上から見た水平面(xy平面)で、
関節1(q1)は肩=原点まわり、関節2(q2)は肘まわりに回る。角度は上から見て反時計回りが正 [rad]。
q2 は「リンク1の向きから見た相対角」であることに注意(絶対角は q1+q2)。
"""
import numpy as np

# リンクの長さ [m]。models/arm2link.xml の capsule の長さと同じ値にしておくこと
# (食い違うと fk の答えが MuJoCo とずれる)。
L1 = 0.2
L2 = 0.15


def fk(q1, q2):
    
    """順運動学: 関節角 (q1, q2) [rad] から、手先の位置を np.array([x, y]) [m] で返す。

    ヒント: 肘の位置 = 肩からリンク1の向き(角度 q1)へ L1 進んだ点。
            手先の位置 = 肘から、リンク2の向き(角度 q1+q2)へ L2 進んだ点。
    """
    x = L1 * np.cos(q1) + L2 * np.cos(q1 + q2)
    y = L1 * np.sin(q1) + L2 * np.sin(q1 + q2)
    return np.array([x, y])
    raise NotImplementedError("fk が未実装です")
    
