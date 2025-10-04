/*
 * Vocab Extractor with Count
 * Developed by Kouki Takashima, Department of Computer Science, Chiba Institute of Technology
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define LIMIT 100000
#define FILE_PATH "/tf/paper/cbow-com/wiki-cleaned.nostopwords.txt"

int main(int argc, char *argv[]) {
        FILE *fp_before, *fp_after;
        char *path = FILE_PATH;
        char target_fpath[200];
        int limit = LIMIT;
        int counter = 1;
        char ch[64];
        int count = 0;

        if(argc == 2) {
                fprintf(stdout, "limit changed: %s\n", argv[1]);
                limit = atoi(argv[1]);
        }

        fp_before = fopen(FILE_PATH, "r");
        if(fp_before == NULL) {
                fprintf(stderr, "Source File %s does not found\n", FILE_PATH);
                return 1;
        }

        sprintf(target_fpath, "wiki-cleaned.%d.vocabc", limit);

        fp_after = fopen(target_fpath, "w");
        if(fp_after == NULL) {
                fprintf(stderr, "Target file could not be opened\n");
                return 2;
        }
        time_t start_time = time(NULL);
        fprintf(stdout, "Task Start. UNIX TIME: %ld\n", start_time);
        while(counter <= limit) {

                fscanf(fp_before, "%s %d", ch, &count);
                fprintf(fp_after, "%s %d\n", ch, count);

                counter++;
                if((counter % 100) == 0) fprintf(stdout, "\r\033[0KProcesing: %d line", counter);
        }

        time_t end_time = time(NULL);
        fprintf(stdout, "\nTask Completed. UNIX TIME: %ld\n", end_time);
        fprintf(stdout, "Duration: %ld\n", end_time - start_time);

        fclose(fp_after);
        fclose(fp_before);
}
