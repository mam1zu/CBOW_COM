#ifndef COMMON_H
#define COMMON_H

/*
 * common.h 
 *
 * Copyright (c) 2014 The Board of Trustees of
 * The Leland Stanford Junior University. All Rights Reserved.
 *
 * License: Apache License 2.0
 *
 * Modified by Kouki Takashima, Department of Computer Science, Chiba Institute of Technology
 *
 */

#include <stdio.h>

#define MAX_STRING_LENGTH 1000
#define TSIZE 1048576
#define SEED 1159241
#define HASHFN bitwisehash

typedef double real;
typedef struct hashrec {
	char *word;
	long long num; //count or id
	struct hashrec *next;
} HASHREC;

int scmp( char *s1, char *s2 );
unsigned int bitwisehash(char *word, int tsize, unsigned int seed);
HASHREC **inithashtable(void);
int get_word(char *word, FILE *fin);
void free_table(HASHREC **ht);
void free_fid(FILE **fid, const int num);

#endif /* COMMON_H */
