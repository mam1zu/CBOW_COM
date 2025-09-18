#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "common.h"

#define SRC_FILE_PATH "/home/kouki-takashima/wiki-cleaned.txt"
#define DST_FILE_PATH "/home/kouki-takashima/wiki-cleaned.100000.txt"
#define VCB_FILE_PATH "/home/kouki-takashima/wiki-cleaned.100000.vocab"
#define UNK_TOKEN "<UNK>"

void hashinsert(HASHREC **ht, char *w) {
    HASHREC     *htmp, *hprv;
    unsigned int hval = HASHFN(w, TSIZE, SEED);

    for (hprv = NULL, htmp = ht[hval]; htmp != NULL && scmp(htmp->word, w) != 0; hprv = htmp, htmp = htmp->next);
    if (htmp == NULL) {
        htmp = (HASHREC *) malloc( sizeof(HASHREC) );
        htmp->word = (char *) malloc( strlen(w) + 1 );
        strcpy(htmp->word, w);
        htmp->num = 1;
        htmp->next = NULL;
        if ( hprv==NULL )
            ht[hval] = htmp;
        else
            hprv->next = htmp;
    }
    else {
        /* new records are not moved to front */
        htmp->num++;
        if (hprv != NULL) {
            /* move to front on access */
            hprv->next = htmp->next;
            htmp->next = ht[hval];
            ht[hval] = htmp;
        }
    }
    return;
}

int main(void) {
	
	FILE *src_fp, *dst_fp, *vcb_fp;
	char ch[200];
	HASHREC **vocab_hash = inithashtable();
	HASHREC *htmp;
	long long i = 0;
	unsigned short flag = 0;
	time_t start_time, end_time;

	src_fp = fopen(SRC_FILE_PATH, "r");

	if(src_fp == NULL) {
		fprintf(stderr, "Source File %s does not found\n", SRC_FILE_PATH);
		return 1;
	}

	vcb_fp = fopen(VCB_FILE_PATH, "r");

	if(vcb_fp == NULL) {
		fprintf(stderr, "Vocabulary File %s does not found\n", VCB_FILE_PATH);
		return 2;
	}

	dst_fp = fopen(DST_FILE_PATH, "w");

	if(dst_fp == NULL) {
		fprintf(stderr, "Destination File %s can't be opened\n", DST_FILE_PATH);
		return 3;
	}

	start_time = time(NULL);
	fprintf(stderr, "Phase 1: Generate Hash Table Using Vocabulary File. UNIX TIME: %ld\n", start_time);
	fprintf(stderr, "Scanning Vocabulary File: %lld words.");
	while( !feof(vcb_fp) ) {
		int nl = get_word(std, src_fp);
		if(nl) continue;
		if(scmp(str, "<UNK>") == 0) {
			fprintf(stderr, "ERROR. <UNK> token detected. please remove.\n");
			free_table(vocab_hash);
			fclose(src_fp);
			fclose(vcb_fp);
			fclose(dst_fp);
			return 4;
		}
		hashinsert(vocab_hash, str);
		if(((++i) % 10000) == 0) fprintf(stderr, "\r\033[0KScanning Vocabulary File: %lld words.", i);
	}


	fprintf(stderr, "\r\033[0KScanning Vocabulary File: %lld words\n", i);
	fprintf(stderr, "Phase 1 Completed.\n");
	fprintf(stderr, "Phase 2: Replacing unlisted tokens with <UNK> token.\n");

	i = 0;
	while(fscanf(src_fp, "%s", ch) != EOF) {
		while(1) {
			htmp = vocab_hash[HASHFN(ch, TSIZE, SEED)];
			if(scmp(htmp->word, ch) == 0) {
				++flag;
				break;
			}
			else if(htmp->next != NULL) {
				htmp = htmp->next;
				continue;
			}
			else break;
		}
		if(flag)
			fprintf(dst_fp, "%s ", ch);
		else
			fprintf(dst_fp, "%s ", UNK_TOKEN);
		
		if(((++i) % 100000) == 0) fprintf(stderr, "\r\033[0KProcessed %lld tokens.", i);
	}
	fprintf(stderr, "\r\033[0KProcessed %lld tokens.\n", i);

	end_time = time(NULL);
	fprintf(stderr, "Phase 2 Completed. UNIX TIME: %ld\n", end_time);
	fprintf(stderr, "Duration: %ld[s]\n", end_time - start_time);
	free_table(vocab_hash);
	fclose(src_fp);
	fclose(dst_fp);

}
