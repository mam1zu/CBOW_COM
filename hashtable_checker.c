#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <locale.h>
#include "common.h"
#include "stdbool.h"

#define SRC_FILE_PATH "/tf/paper/wikidata/wiki-cleaned.txt"
#define DST_FILE_PATH "/tf/paper/cbow-com/wiki-cleaned.100000.txt.test"
#define VCB_FILE_PATH "/tf/paper/cbow-com/utils/wiki-cleaned.nostopword.100000.vocab"
#define UNK_TOKEN "<UNK>"
#define VOCAB_SIZE 100001

int word_counter = 1;
char **id_to_word;

HASHREC *hashsearch(HASHREC **ht, char *w) {
	HASHREC *htmp, *hprv;
	unsigned int hval = HASHFN(w, TSIZE, SEED);
	for(hprv = NULL, htmp=ht[hval]; htmp != NULL && scmp(htmp->word, w) != 0; hprv = htmp, htmp= htmp->next);
	if(htmp != NULL && hprv != NULL) {
		hprv->next = htmp->next;
		htmp->next = ht[hval];
		ht[hval] = htmp;
	}
	return htmp;
}

void hashinsert(HASHREC **ht, char *w) {
    HASHREC     *htmp, *hprv;
    unsigned int hval = HASHFN(w, TSIZE, SEED);

    for (hprv = NULL, htmp = ht[hval]; htmp != NULL && scmp(htmp->word, w) != 0; hprv = htmp, htmp = htmp->next);
    if (htmp == NULL) {
        htmp = (HASHREC *) malloc( sizeof(HASHREC) );
        htmp->word = (char *) malloc( strlen(w) + 1 );
        strcpy(htmp->word, w);
        htmp->num = word_counter;
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
	
	FILE *vcb_fp;
	char ch[MAX_STRING_LENGTH];
	HASHREC **vocab_hash = inithashtable();
	HASHREC *htmp;
	long long i = 0, hoge = 0;
	int mode = 0, id = 0;
	long long counter = 0;
	unsigned short flag = 0;
	time_t start_time, end_time;

	vcb_fp = fopen(VCB_FILE_PATH, "r");

	if(vcb_fp == NULL) {
		fprintf(stderr, "Vocabulary File %s not found\n", VCB_FILE_PATH);
		return 2;
	}

	id_to_word = (char **)malloc(sizeof(char *) * VOCAB_SIZE);
	for(hoge = 0; hoge < VOCAB_SIZE; hoge++)
		id_to_word[hoge] = (char *)malloc(sizeof(char) * 16);
	
	start_time = time(NULL);
	fprintf(stderr, "Phase 1: Generate Hash Table Using Vocabulary File. UNIX TIME: %ld\n", start_time);
	fprintf(stderr, "Scanning Vocabulary File: %lld words.", i);
	while( !feof(vcb_fp) ) {
		int nl = get_word(ch, vcb_fp);
		if(nl) continue;
		if(scmp(ch, "<UNK>") == 0) {
			fprintf(stderr, "ERROR. <UNK> token detected. please remove.\n");
			free_table(vocab_hash);
			fclose(vcb_fp);
			return 4;
		}
		hashinsert(vocab_hash, ch);
		strcpy(id_to_word[word_counter], ch);
		fprintf(stderr, "%d : %s\n", word_counter, id_to_word[word_counter]);
		word_counter++;
		if(((++i) % 10000) == 0) fprintf(stderr, "\r\033[0KScanning Vocabulary File: %lld words.", i);
	}


	fprintf(stderr, "\r\033[0KScanning Vocabulary File: %lld words\n", i);
	fprintf(stderr, "Phase 1 Completed.\n");

	while(true) {
		if(mode == 0) {
			fprintf(stderr, "Enter mode. 1: word_to_id, 2: id_to_word");
			fscanf(stdin, "%d", &mode);
			if(mode != 1 && mode != 2) {
				mode = 0;
			}
			continue;
		}
		if(mode == 1) {
			fprintf(stderr, "input words to check id: ");
                	fscanf(stdin, "%s", ch);
                	if(strcmp(ch, "NANIWOSURUNOKA") == 0) {
                        	fprintf(stderr, "BYE\n");
                   		break;
                	}

                	htmp = hashsearch(vocab_hash, ch);
                	if(htmp != NULL)
                        	fprintf(stderr, "%lld\n", htmp->num);
                	else
                        	fprintf(stderr, "NULL\n");
		}

		if(mode == 2) {
			fprintf(stderr, "input id to check word: ");
			fscanf(stdin, "%s", ch);
			id = atoi(ch);
			if(id == 2231081) {
				fprintf(stderr, "BYE\n");
				break;
			}
			if(0 <= id && id <= VOCAB_SIZE) {
				fprintf(stderr, "%s\n", id_to_word[id]);
			}
			else {
				fprintf(stderr, "NOPE\n");
			}
		}
	}
	

	free_table(vocab_hash);
	free(id_to_word);
	fclose(vcb_fp);

}
