#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "common.h"

#define SRC_FILE_PATH "/home/kouki-takashima/wiki-cleaned.txt"
#define DST_FILE_PATH "/home/kouki-takashima/wiki-cleaned.100000.txt""
#define VCB_FILE_PATH "/home/kouki-takashima/wiki-cleaned.100000.vocab"

void hashinsert(HASHREC **ht, char *w) {
	HASHREC     *htmp, *hprev;
	unsigned int hval = HASHFN(w, TSIZE, SEED);

	for(hprv = NULL, htmp = ht[hval]; htmp != NULL && scmp(htmp->word, w) != 0; hprv = htmp, htmp = htmp->next);
	if(htmp == NULL) {
		htmp = (HASHREC *)malloc(sizeof(HASHREC));
		htmp->word = (char *)malloc(strlen(w) +1);
	}
}

int main(void) {
	
	FILE *src_fp, *dst_fp, *vcb_fp;
	char ch[200];
	HASHREC **vocab_hash = inithashtable();
	HASHREC *htmp;

	src_fp = fopen(SRC_FILE_PATH, 'r');

	if(src_fp == NULL) {
		fprintf(stderr, "Source File %s does not found\n", SRC_FILE_PATH);
		return 1;
	}

	vcb_fp = fopen(VCB_FILE_PATH, 'r');

	if(vcb_fp == NULL) {
		fprintf(stderr, "Vocabulary File %s does not found\n", VCB_FILE_PATH);
		return 2;
	}

	dst_fp = fopen(DST_FILE_PATH, 'w');

	if(dst_fp == NULL) {
		fprintf(stderr, "Destination File %s can't be opened\n", DST_FILE_PATH);
		return 3;
	}

	

	while(fscanf(src_fp, "%s", ch) != EOF) {

	}



	fclose(src_fp);
	fclose(dst_fp);

}
