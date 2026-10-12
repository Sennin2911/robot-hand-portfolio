# SO-101 モデル(MuJoCo)

- 出典: https://github.com/TheRobotStudio/SO-ARM100 の `Simulation/SO101/`
  (コミット a758567c3978dfeefe282ede0500085a48fe8f78 時点)
- ライセンス: Apache-2.0(`LICENSE` を同梱)
- 取り込んだもの: `so101_new_calib.xml`(本体)、`scene.xml`(床と照明)、`assets/` のうち本体が使うメッシュ13個のみ。
  リーダーアーム・URDF・旧キャリブレーション版・カメラ付き版は取り込んでいない。
- 変更: なし(原本のまま)。
- キャリブレーション: new_calib = 各関節の 0 が可動域の中央。LeRobot の現行キャリブレーションに対応する想定
  (実機到着後に要確認)。
- 注意: LeRobot ではグリッパを 0〜100 の直動量として扱うが、このモデルでは回転関節(rad)。変換は未実装。
