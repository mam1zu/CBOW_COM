#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "common.h"

#define LIMIT 100000
#define ALPHA 0.75
#define VCBC_FILE_PATH "/tf/paper/cbow-com/clang-utils/wiki-cleaned.nostopword.100000.vocabc"
#define DST_FILE_PATH "/tf/paper/cbow-com/clang-utils/wiki-cleaned.nostopword.100000.negdist"
#define MODE 0 //MODE 0: 単語, 確率をテキスト形式で出力, 1: 配列長100000の確率をnumpy形式で出力


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


int write_header_reserve(FILE *fin, int byte) {
        if(fin == NULL) return -1;
        int counter;
        char zero_char = '\x0';
        for(counter = 0; counter < byte; counter++) fwrite(&zero_char, sizeof(zero_char), 1, fin);
        return counter;
}

int write_npy_header(FILE *fin, char dtype[], long long *shape, int dim) {

        long long i;
        char magic_word[6] = "\x93NUMPY"; //マジックワード、6バイト固定
        unsigned char version_major = '\x01'; //1.
        unsigned char version_minor = '\x00'; //0
        unsigned short len_whole, len_header;
        char buf[256];
        char header_1[65536];
        char *header_2;

        memset(header_1, 0, sizeof(header_1));
        memset(buf, 0, sizeof(buf));
	sprintf(header_1, "{'descr': '<%s', 'fortran_order': False, 'shape': (", dtype);

        if(fin == NULL) {
                //FILE POINTER IS NULL, ERRNO: 1
                return 1;
        }

        if(shape == NULL) {
                //SHAPE POINTER IS NULL, ERRNO: 2
                return 2;
        }

        for(i = 0; i < dim; ++i) {
                sprintf(buf, "%lld", shape[i]);
                strcat(header_1, buf);
                strcat(header_1, ", ");
                //最後以外ならコンマを入れる
        }
        strcat(header_1, "), }");

        //ヘッダサイズの計算
        len_header = strlen(header_1) * sizeof(char);
        len_whole = sizeof(magic_word) + sizeof(version_major) + sizeof(version_minor)
                        + sizeof(len_header) + len_header;
        memset(buf, 0, sizeof(buf));
        for(;len_whole % 64 != 0; ++len_whole) {
                //メモリ確保前にアラインメントサイズを計算して、それからメモリを確保する
                strcat(buf, "\x20"); //先にバッファにパディングデータを書き込ん でおく、segfaultが怖い
                ++len_header;
        }

        //メモリ確保、ただしメタデータの大きさを考慮する
        header_2 = (char *)malloc(len_whole);
        strcpy(header_2, header_1); //デカバッファの1から確保したぴったりサイズ の2に移行
        strcat(header_2, buf); //パディングデータをバッファから読み込んで連結
        /*
                下の行のコード、わかりずらく多分忘れるのでメモ
                最後の要素を改行コードに置き換える。
                header_2の最後の要素を示すインデックスはstrlen(header_2) - 1である。
                例: アラインメントで128となっている場合、最後のバイトを示すイン デックスは127。
                これをmemmoveを用いて置き換える。memcpyでもいいがmemmoveのほうが安全なのでこっちにしてみる
        */
        memmove(&header_2[strlen(header_2) - 1], "\n", sizeof(char));


        fwrite(magic_word, sizeof(magic_word), 1, fin);        //1. マジックワード 6byte (char *)
        fwrite(&version_major, sizeof(version_major), 1, fin); //2. メジャーバージョン 1byte (unsigned char)
        fwrite(&version_minor, sizeof(version_minor), 1, fin); //3. マイナーバージョン 1byte (unsigned char)
        fwrite(&len_header, sizeof(len_header), 1, fin);                     //4. ヘッダサイズ 4byte (unsigned short)
        fwrite(header_2, len_header, 1, fin);                         //5. ヘッ ダ len byte (char *)

        free(header_2);
        return 0;

}

int main(int argc, char *argv[]) {

        FILE *vcbc_fp, *dst_fp;
        char ch[256], dtype[] = "f4", dst_filename[256];
        int limit = LIMIT, dim = 1, mode;
        long long shape[1] = {limit}; //ポインタである必要があるため要素1でも配 列に
        int counter, i, j, k, *count;
        float tmp = 0.0f, distrib;
        char **words;

	if(argc != 2) {
                fprintf(stderr, "Incorrect parameters, enter mode 0 (output txt) or 1 (output npy)\n");
                return 1;
        }
        switch(atoi(argv[1])) {
                case 0:
			mode = 0;
			break;
		case 1:
			mode = 1;
			break;
		default:
			fprintf(stderr, "Incorrect mode number. Enter 0 or 1.\n");
			return 1;
        }


        count = (int *)malloc(sizeof(int) * LIMIT);
        words = (char **)malloc(sizeof(char *) * LIMIT);
        for(i = 0; i < LIMIT; i++)
                words[i] = (char *)malloc(sizeof(char) * 256);

        sprintf(dst_filename, "%s.%s", DST_FILE_PATH, (mode == 0? "txt" : "npy"));
        if((vcbc_fp = fopen(VCBC_FILE_PATH, "r")) == NULL) {
                fprintf(stderr, "Vocabulary File (with word count) %s not found\n", VCBC_FILE_PATH);
                return 2;
        }

        if((dst_fp = fopen(dst_filename, (mode == 0 ? "w" : "wb" ))) == NULL) {
                fprintf(stderr, "Destination File %s not found\n", dst_filename);
                return 3;
        }


        for(i = 0; i < LIMIT; i++) {
                fscanf(vcbc_fp, "%s %d", words[i], &counter);
                count[i] = counter;
                tmp += (float)counter;
		fprintf(stderr, "%d ", count[i]);
        }

	tmp = pow(tmp, ALPHA);

        if(mode == 0) {
                for(i = 0; i < LIMIT; i++) {
			distrib = pow((float)count[i], ALPHA) / tmp;
                        fprintf(dst_fp, "%s %f\n", words[i], distrib);
                }
        }

        else {
                // MODE == 1
                write_npy_header(dst_fp, dtype, shape, dim);

                for(i = 0; i < LIMIT; i++) {
                        distrib = pow((float)count[i], ALPHA) / tmp;
                        fwrite(&distrib, sizeof(distrib), 1, dst_fp);
                }
        }

	free(count);

	for(i = 0; i < LIMIT; i++)
		free(words[i]);

	free(words);

        return 0;
}
