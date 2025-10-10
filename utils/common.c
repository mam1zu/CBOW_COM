/*
 * common.c
 *
 * Copyright (c) 2014 The Board of Trustees of
 * The Leland Stanford Junior University. All Rights Reserved.
 *
 * License: Apache License 2.0
 *
 * Modified by Kouki Takashima, Department of Computer Science, Chiba Institute of Technology.
 * 
 */

#include <stdlib.h>
#include <string.h>
#include "common.h"

/* Efficient string comparison */
int scmp( char *s1, char *s2 ) {
	while (*s1 != '\0' && *s1 == *s2) { s1++; s2++;}
	return (*s1 - *s2);
}

unsigned int bitwisehash(char *word, int tsize, unsigned int seed) {
	char c;
	unsigned int h;
	h = seed; //Allocate randomly and uses the value as a seed
	for (; (c = *word) != '\0'; word++)
		h ^= ((h << 5) + c + (h >> 2));
	return (unsigned int)((h & 0x7fffffff) % tsize);

}

/* Create hasht table, initialise pointers to NULL*/

HASHREC **inithashtable(void) {
	int i;
	HASHREC **ht;
	ht = (HASHREC **) malloc( sizeof(HASHREC *) * TSIZE);
	for (i = 0; i < TSIZE; i++)
		ht[i] = (HASHREC *) NULL;//Not allocated at that time, but assigned NULL
	return ht;
}

int get_word(char *word, FILE *fin) {
    int i = 0, ch;
    for ( ; ; ) {
        ch = fgetc(fin);
        if (ch == '\r') continue;
        if (i == 0 && ((ch == '\n') || (ch == EOF))) {
            word[i] = 0;
            return 1;
        }
        if (i == 0 && ((ch == ' ') || (ch == '\t'))) continue; // skip leading space
        if ((ch == EOF) || (ch == ' ') || (ch == '\t') || (ch == '\n')) {
            if (ch == '\n') ungetc(ch, fin); // return the newline next time as document ender
            break;
        }
        if (i < MAX_STRING_LENGTH - 1)
          word[i++] = ch; // don't allow words to exceed MAX_STRING_LENGTH
    }
    word[i] = 0; //null terminate
    if (i == MAX_STRING_LENGTH - 1 && (word[i-1] & 0x80) == 0x80) {
        if ((word[i-1] & 0xC0) == 0xC0) {
            word[i-1] = '\0';
        } else if (i > 2 && (word[i-2] & 0xE0) == 0xE0) {
            word[i-2] = '\0';
        } else if (i > 3 && (word[i-3] & 0xF8) == 0xF0) {
            word[i-3] = '\0';
        }
    }
    return 0;
}

int find_arg(char *str, int argc, char **argv) {
    int i;
    for (i = 1; i < argc; i++) {
        if (!scmp(str, argv[i])) {
            if (i == argc - 1) {
                printf("No argument given for %s\n", str);
                exit(1);
            }
            return i;
        }
    }
    return -1;
}

void free_table(HASHREC **ht) {
	int i;
	HASHREC *current;
	HASHREC *tmp;

	for(i = 0; i < TSIZE; i++) {
		current = ht[i];
		while (current != NULL) {
			tmp = current;
			current = current->next;
			free(tmp->word);
			free(tmp);
		}
	}
	free(ht);
}

void free_fid(FILE **fid, const int num) {
	int i;
	for(i = 0; i < num; i++) {
		if(fid[i] != NULL)
			fclose(fid[i]);
	}
	free(fid);
}


