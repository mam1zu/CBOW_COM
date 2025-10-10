import sys
from tqdm import tqdm

SRC_FILE_PATH = "/tf/paper/wikidata/wiki-cleaned.txt"
DST_FILE_PATH = "/tf/paper/cbow-com/wiki-cleaned.nostopword.txt"

ENGLISH_STOP_WORDS = [
    "a", "about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around",
    "as", "at", "back", "be", "became", "because", "become", "becomes",
    "becoming", "been", "before", "beforehand", "behind", "being", "below",
    "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but",
    "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt",
    "cry", "de", "describe", "detail", "do", "done", "down", "due", "during",
    "each", "eg", "either", "eleven", "else", "elsewhere", "empty",
    "enough", "etc", "even", "ever", "every", "everyone", "everything",
    "everywhere", "except", "few", "fifteen", "fifty", "fill", "find", "fire",
    "first", "for", "former", "formerly", "forty", "found",
    "from", "front", "full", "further", "get", "give", "go", "had", "has",
    "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby",
    "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how",
    "however", "i", "ie", "if", "in", "inc", "indeed", 
    "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly",
    "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might",
    "mill", "mine", "more", "moreover", "most", "mostly", "move", "much",
    "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless",
    "next", "no", "nobody", "none", "noone", "nor", "not", "nothing",
    "now", "nowhere", "of", "off", "often", "on", "once", "only",
    "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves",
    "out", "over", "own", "part", "per", "perhaps", "please", "put", "rather",
    "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious",
    "several", "she", "should", "show", "side", "since", "sincere",
    "sixty", "so", "some", "somehow", "someone", "something", "sometime",
    "sometimes", "somewhere", "still", "such", "system", "take", "than",
    "that", "the", "their", "them", "themselves", "then", "thence", "there",
    "thereafter", "thereby", "therefore", "therein", "thereupon", "these",
    "they", "thick", "thin", "third", "this", "those", "though",
    "through", "throughout", "thru", "thus", "to", "together", "too", "top",
    "toward", "towards", "twelve", "twenty", "un", "under", "until",
    "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what",
    "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas",
    "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while",
    "whither", "who", "whoever", "whole", "whom", "whose", "why", "will",
    "with", "within", "without", "would", "yet", "you", "your", "yours",
    "yourself", "yourselves", "st", "nd", "rd", "th"
]

def remove_stopwords_from_corpus(corpus_path, output_path):

    stopwords_set = frozenset(ENGLISH_STOP_WORDS)
    print(f"Generated a hashtable (set) of {len(stopwords_set)} stopwords.", file=sys.stderr)
    
    try:
        with open(corpus_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            print("Scanning and removing stop words from corpus...", file=sys.stderr)
            
            # tqdmを使って進捗バーを表示
            for line in tqdm(infile, desc="Processing lines"):
                # 行を単語に分割
                words = line.split()
                # ストップワードでない単語だけを抽出
                filtered_words = [word for word in words if word not in stopwords_set]
                # 新しい行をスペース区切りで再構築
                new_line = ' '.join(filtered_words)
                # 出力ファイルに書き込む
                outfile.write(new_line + '\n')
                
    except FileNotFoundError:
        print(f"Error: Corpus file not found at '{corpus_path}'", file=sys.stderr)
        sys.exit(1)
        
    print(f"\nDone. Output saved to '{output_path}'", file=sys.stderr)

def main():
    """メイン実行関数"""
    remove_stopwords_from_corpus(SRC_FILE_PATH, DST_FILE_PATH)

if __name__ == '__main__':
    main()