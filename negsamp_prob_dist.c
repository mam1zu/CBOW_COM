#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <locale.h>
#include "common.h"
#include "stdbool.h"

#define VCB_FILE_PATH "/tf/paper/cbow-com/wiki-cleaned.100000.vocab"
#define UNK_TOKEN "<UNK>"

int main(void) {
    FILE *vcb_fp;

    vcb_fp = fopen(VCB_FILE_PATH, "r");

    if(vcb_fp == NULL) {
        fprintf(stderr, "Vocabulary File %s not found\n", VCB_FILE_PATH);
        return 1;
    }


}