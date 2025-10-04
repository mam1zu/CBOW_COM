#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "common.h"

//このプログラムが必要とするもの
//1. コーパスファイル
//2. コーパスファイルの全語彙を含んだvocabファイル

//このプログラムができるようにすること
//1. ストップワードの除去(上位何個、などの決め方)
//2. vocabファイルから語彙数をカット
//3. 語彙数をカットしたvocabファイルを用いてコーパスの語彙数をカット

#define SRC_FILE_PATH ""
#define VCB_FILE_PATH ""
#define DST_FILE_PATH ""
#define STOP_WORD_NUM 100
#define VOCAB_SIZE 100000

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


void stopword_remover(FILE *vcb_fp, FILE *src_fp, char *dst_file_path) {

    fprintf(stderr, "stopword_remover start");

    FILE *dst_fp;
	HASHREC **stopwords_ht = inithashtable();
	HASHREC *htmp;
	int flag, id = 1;
	long long i;
	char ch[256];

	if((dst_fp = fopen(dst_file_path, "w")) == NULL) {
		fprintf(stderr, "Destination File %s can't be opened\n", dst_file_path);
		return 1;
	}

	fprintf(stderr, "Generating hashtable of stopwords only\n");

	for(i = 0; i < STOP_WORD_NUM; i++) {
		fscanf(vcb_fp, "%s", ch);
		hashinsert(stopwords_ht, ch, id);
		id++;
	}

	fprintf(stderr, "Done\n");

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

		if((++i) % 100000 == 0) fprintf(stderr, "\r\033[0KProcessed %lld token.", i);
	}

	fprintf(stderr, "\r\033[0KProcessed %lld token\n", i);
	fprintf(stderr, "Done\n");

	free_table(stopwords_ht);

    fprintf(stderr, "stopword_remover finished");
    return 0;

}

int vocab_extractor(FILE *fp_before) {
    FILE *fp_after;
    char target_fpath[200];
    int vocab_size = VOCAB_SIZE;
    int counter = 1;
    char ch[64];
    int count = 0;

    if(fp_before == NULL) return 1;

    fprintf(stderr, "vocab_extractor start\n");

    sprintf(target_fpath, "wiki-cleaned.%d.vocabc", vocab_size);

    fp_after = fopen(target_fpath, "w");
    if(fp_after == NULL) {
            fprintf(stderr, "Target file could not be opened\n");
            return 2;
    }

    while(counter <= vocab_size) {

            fscanf(fp_before, "%s %d", ch, &count);
            fprintf(fp_after, "%s %d\n", ch, count);

            counter++;
            if((counter % 100) == 0) fprintf(stdout, "\r\033[0KProcesing: %d line", counter);
    }

    fprintf(stdout, "\r\033[0KProcesing: %d line\n", counter);

    fclose(fp_after);

    fprintf(stderr, "vocab_extractor finished\n");
    return 0;

}

int main(void) {
    return;
}