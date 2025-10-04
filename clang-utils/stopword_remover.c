#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "common.h"

#define VOCAB_SIZE 100000
#define STOP_WORD_NUM 318
#define VCB_FILE_PATH "/tf/paper/cbow-com/wiki-cleaned.100000.vocab"
#define SRC_FILE_PATH "/tf/paper/wikidata/wiki-cleaned.txt"
#define DST_FILE_PATH "/tf/paper/cbow-com/wiki-cleaned.nostopword.txt"

char *english_stop_words[] = {
        "a",
        "about",
        "above",
        "across",
        "after",
        "afterwards",
        "again",
        "against",
        "all",
        "almost",
        "alone",
        "along",
        "already",
        "also",
        "although",
        "always",
        "am",
        "among",
        "amongst",
        "amoungst",
        "amount",
        "an",
        "and",
        "another",
        "any",
        "anyhow",
        "anyone",
        "anything",
        "anyway",
        "anywhere",
        "are",
        "around",
        "as",
        "at",
        "back",
        "be",
        "became",
        "because",
        "become",
        "becomes",
        "becoming",
        "been",
        "before",
        "beforehand",
        "behind",
        "being",
        "below",
        "beside",
        "besides",
        "between",
        "beyond",
        "bill",
        "both",
        "bottom",
        "but",
        "by",
        "call",
        "can",
        "cannot",
        "cant",
        "co",
        "con",
        "could",
        "couldnt",
        "cry",
        "de",
        "describe",
        "detail",
        "do",
        "done",
        "down",
        "due",
        "during",
        "each",
        "eg",
        "eight",
        "either",
        "eleven",
        "else",
        "elsewhere",
        "empty",
        "enough",
        "etc",
        "even",
        "ever",
        "every",
        "everyone",
        "everything",
        "everywhere",
        "except",
        "few",
        "fifteen",
        "fifty",
        "fill",
        "find",
        "fire",
        "first",
        "five",
        "for",
        "former",
        "formerly",
        "forty",
        "found",
        "four",
        "from",
        "front",
        "full",
        "further",
        "get",
        "give",
        "go",
        "had",
        "has",
        "hasnt",
        "have",
        "he",
        "hence",
        "her",
        "here",
        "hereafter",
        "hereby",
        "herein",
        "hereupon",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "however",
        "hundred",
        "i",
        "ie",
        "if",
        "in",
        "inc",
        "indeed",
        "interest",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "keep",
        "last",
        "latter",
        "latterly",
        "least",
        "less",
        "ltd",
        "made",
        "many",
        "may",
        "me",
        "meanwhile",
        "might",
        "mill",
        "mine",
        "more",
        "moreover",
        "most",
        "mostly",
        "move",
        "much",
        "must",
        "my",
        "myself",
        "name",
        "namely",
        "neither",
        "never",
        "nevertheless",
        "next",
        "nine",
        "no",
        "nobody",
        "none",
        "noone",
        "nor",
        "not",
        "nothing",
        "now",
        "nowhere",
        "of",
        "off",
        "often",
        "on",
        "once",
        "one",
        "only",
        "onto",
        "or",
        "other",
        "others",
        "otherwise",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "part",
        "per",
        "perhaps",
        "please",
        "put",
        "rather",
        "re",
        "same",
        "see",
        "seem",
        "seemed",
        "seeming",
        "seems",
        "serious",
        "several",
        "she",
        "should",
        "show",
        "side",
        "since",
        "sincere",
        "six",
        "sixty",
        "so",
        "some",
        "somehow",
        "someone",
        "something",
        "sometime",
        "sometimes",
        "somewhere",
        "still",
        "such",
        "system",
        "take",
        "ten",
        "than",
        "that",
        "the",
        "their",
        "them",
        "themselves",
        "then",
        "thence",
        "there",
        "thereafter",
        "thereby",
        "therefore",
        "therein",
        "thereupon",
        "these",
        "they",
        "thick",
        "thin",
        "third",
        "this",
        "those",
        "though",
        "three",
        "through",
        "throughout",
        "thru",
        "thus",
        "to",
        "together",
        "too",
        "top",
        "toward",
        "towards",
        "twelve",
        "twenty",
        "two",
        "un",
        "under",
        "until",
        "up",
        "upon",
        "us",
        "very",
        "via",
        "was",
        "we",
        "well",
        "were",
        "what",
        "whatever",
        "when",
        "whence",
        "whenever",
        "where",
        "whereafter",
        "whereas",
        "whereby",
        "wherein",
        "whereupon",
        "wherever",
        "whether",
        "which",
        "while",
        "whither",
        "who",
        "whoever",
        "whole",
        "whom",
        "whose",
        "why",
        "will",
        "with",
        "within",
        "without",
        "would",
        "yet",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
};

HASHREC *hashsearch(HASHREC **ht, char *w) {
	HASHREC *htmp, *hprv;
	unsigned int hval = HASHFN(w, TSIZE, SEED);
	for (hprv = NULL, htmp=ht[hval]; htmp != NULL && scmp(htmp->word, w) != 0; hprv = htmp, htmp = htmp->next);
	if(htmp != NULL && hprv != NULL) {
		hprv->next = htmp->next;
		htmp->next = ht[hval];
		ht[hval] = htmp;
	}
	return(htmp);
}

void hashinsert(HASHREC **ht, char *w, long long id) {
	HASHREC *htmp, *hprv;
	unsigned int hval = HASHFN(w, TSIZE, SEED);
	for(hprv = NULL, htmp = ht[hval]; htmp != NULL && scmp(htmp->word, w) != 0; hprv = htmp, htmp = htmp->next);
	if(htmp == NULL) {
		htmp = (HASHREC *)malloc(sizeof(HASHREC));
		htmp->word = (char *)malloc(strlen(w) + 1);
		strcpy(htmp->word, w);
		htmp->num = id;
		htmp->next = NULL;
		if(hprv == NULL) ht[hval] = htmp;
		else hprv->next = htmp;
	}
	else
		fprintf(stderr, "Error, duplicate entry located: %s.\n", htmp->word);
	return;
}

int main(void) {
	FILE *vcb_fp, *src_fp, *dst_fp;
	HASHREC **stopwords_ht = inithashtable();
	HASHREC *htmp;
	int flag, id = 1;
	long long i;
	char ch[256];
	// if((vcb_fp = fopen(VCB_FILE_PATH, "r")) == NULL) {
	// 	fprintf(stderr, "Vocabulary File %s not found\n", VCB_FILE_PATH);
	// 	return 1;
	// }

	if((src_fp = fopen(SRC_FILE_PATH, "r")) == NULL) {
		fprintf(stderr, "Corpus File %s not found\n", SRC_FILE_PATH);
		return 2;
	}

	if((dst_fp = fopen(DST_FILE_PATH, "w")) == NULL) {
		fprintf(stderr, "Destination File %s can't be opened\n", DST_FILE_PATH);
		return 3;
	}

	fprintf(stderr, "Generating hashtable of stopwords only\n");

	for(i = 0; i < STOP_WORD_NUM; i++) {
		hashinsert(stopwords_ht, english_stop_words[i], id);
		id++;
	}

	fprintf(stderr, "Done. Stop word count: %d\n", id);

	fprintf(stderr, "Scanning and Removing stop words from corpus.");
	i = 0;
	fprintf(stderr, "Processed %lld token.", i);
	while(1) {

		flag = get_word(ch, src_fp);
		if(flag == 1) {
			if(feof(src_fp)) break;
			else {
				//space
				fprintf(dst_fp, "\n");
				continue;
			}
		}

		htmp = hashsearch(stopwords_ht, ch);
		if(htmp != NULL) continue; //stop word detected
		fprintf(dst_fp, "%s ", ch);

		if((i++) % 100000 == 0) fprintf(stderr, "\r\033[0KProcessed %lld token.", i);
	}

	fprintf(stderr, "\r\033[0KProcessed %lld token\n", i);
	fprintf(stderr, "Done\n");

	//fclose(vcb_fp);
	fclose(src_fp);
	fclose(dst_fp);
	free_table(stopwords_ht);

}
