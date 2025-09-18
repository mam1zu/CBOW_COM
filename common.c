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
	int i = 0;
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


