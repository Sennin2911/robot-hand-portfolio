# プロジェクトのルール

## コード作成時のルール (必ず守る)
作者は MuJoCo も Python も初心者 (`import` が分かる程度)。コードは「読んで学べる」ことを優先する。

1. **ファイル先頭に概要コメント**を書く (そのファイルが何をするか、1〜数行)。
2. **初心者が意味を知らなそうな記法・関数には、その行の近くに意味を書く。**
   例: `sys.path.insert`, `Path(__file__).resolve().parents[n]`, `@property`, `self`, `with ... as`,
   スライス (`[1:]`, `[-1]`), f文字列, ジェネレータ式, `os.environ.setdefault`, `try/except` など。
3. **使用頻度もコメントする**: その書き方が「定番でよく使う」のか「あまり使わない/特殊な事情で使っている」のか。
4. **順序に理由があるコード** (例: `mujoco` の import より前に環境変数を設定) は、その理由を明記する。
5. MuJoCo 固有の概念 (model/data, qpos, ctrl, actuator, body, 単位など) も初出で説明する。
6. コメントは日本語。コードの変更時はコメントも更新し、実態とずれた説明を残さない。
7. 自明なこと (`i += 1  # 1足す` のようなもの) は書かない。「初心者が詰まりそうな所」に絞る。

## 環境
- Python 仮想環境は `.venv/` (pip なしで作成済み)。実行は `.venv/bin/python ...`。
- パッケージ追加: `pip3 --python .venv/bin/python install <pkg>` し、`requirements.txt` にも追記する。
- ヘッドレス描画は `MUJOCO_GL=egl`。
