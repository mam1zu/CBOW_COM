#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <locale.h>
#include "common.h"

#define SRC_FILE_PATH "/tf/paper/cbow-com/wiki-cleaned.nostopword.txt"
#define DST_FILE_PATH "/tf/paper/cbow-com/wiki-cleaned.nostopword.100000.txt"
#define VCB_FILE_PATH "/tf/paper/cbow-com/wiki-cleaned.nostopword.100000.vocab"
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
	char ch[MAX_STRING_LENGTH];
	HASHREC **vocab_hash = inithashtable();
	HASHREC *htmp;
	long long i = 0;
	long long counter = 0;
	unsigned short flag = 0;
	time_t start_time, end_time;

	setlocale(LC_ALL, "");

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
	fprintf(stderr, "Scanning Vocabulary File: %lld words.", i);
	while( !feof(vcb_fp) ) {
		int nl = get_word(ch, vcb_fp);
		if(nl) continue;
		if(scmp(ch, "<UNK>") == 0) {
			fprintf(stderr, "ERROR. <UNK> token detected. please remove.\n");
			free_table(vocab_hash);
			fclose(src_fp);
			fclose(vcb_fp);
			fclose(dst_fp);
			return 4;
		}
		hashinsert(vocab_hash, ch);
		if(((++i) % 10000) == 0) fprintf(stderr, "\r\033[0KScanning Vocabulary File: %lld words.", i);
	}


	fprintf(stderr, "\r\033[0KScanning Vocabulary File: %lld words\n", i);
	fprintf(stderr, "Phase 1 Completed.\n");
	fprintf(stderr, "Phase 2: Removing unlisted words from corpus.\n");

	i = 0;
	fprintf(stderr, "Processed %lld tokens.", i);
	while( !feof(src_fp) ) {
		int nl = get_word(ch, src_fp);
		if(nl) {
			fprintf(dst_fp, "\n");
			continue;
		}
		htmp = vocab_hash[HASHFN(ch, TSIZE, SEED)];
		if(htmp == NULL) continue;
		while(1) {
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

		if(flag) {
			--flag;;
			fprintf(dst_fp, "%s ", ch);
		}
		
		if(((++i) % 100000) == 0) fprintf(stderr, "\r\033[0KProcessed %lld tokens.", i);
	}
	fprintf(stderr, "\r\033[0KProcessed %lld tokens.\n", i);

	end_time = time(NULL);
	fprintf(stderr, "Phase 2 Completed. UNIX TIME: %ld\n", end_time);
	fprintf(stderr, "Duration: %ld[s]\n", end_time - start_time);
	free_table(vocab_hash);
	fclose(src_fp);
	fclose(vcb_fp);
	fclose(dst_fp);

}
