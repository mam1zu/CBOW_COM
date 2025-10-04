import numpy as np

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# cbow_old.pyで「実際に読み込んでいるパス」と完全に一致させてください
dataset_path = "/tf/paper/cbow-com/dataset/wiki-cleaned.100000.train.0000"
# cbow_old.pyで「実際にモデルに渡しているvocab_size」と完全に一致させてください
vocab_size = 100001
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

try:
    dataset_0 = np.load(dataset_path)

    min_id = np.min(dataset_0)
    max_id = np.max(dataset_0)

    print(f"--- データ検証スクリプト ---")
    print(f"検証ファイル: {dataset_path}")
    print(f"設定された語彙サイズ: {vocab_size} (有効なID: 0 ～ {vocab_size - 1})")
    print(f"データ内の最小ID: {min_id}")
    print(f"データ内の最大ID: {max_id}")
    print("--------------------------")

    error_found = False
    if min_id < 0:
        print(f"エラー: 負のID ({min_id}) が見つかりました。")
        error_found = True
    
    if max_id >= vocab_size:
        print(f"エラー: 範囲外のID ({max_id}) が見つかりました。")
        error_found = True

    if not error_found:
        print("✔︎ 検証完了: データセット内のすべてのIDは有効な範囲内です。")
    else:
        print("\n再度cbow_old.pyのパスや変数定義が一致しているか見直してください。")

except FileNotFoundError:
    print(f"エラー: ファイルが見つかりません。パスが正しいか確認してください: {dataset_path}")