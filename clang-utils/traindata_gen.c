#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "common.h"

#define DEBUG 0
#define VOCAB_SIZE 100000
#define EMB_DIM 300
#define WINDOW_SIZE 10
#define CORPUS_FILE_PATH "/tf/paper/cbow-com/wiki-cleaned.nostopword.100000.txt"
#define TARGET_DIRECTORY_PATH "/tf/paper/cbow-com/dataset/window_10/"
#define VCB_FILE_PATH "/tf/paper/cbow-com/clang-utils/wiki-cleaned.nostopword.100000.vocab"

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

int write_npy(FILE *fin, long long *shape, int dim, long long *val) {

	long long i;
	if(fin == NULL) {
		return 1;
	}

	if(shape == NULL) {
		return 2;
	}

	if(val == NULL) {
		return 3;
	}
	for(i = 0; i < dim; ++i) {
		fwrite(&val[i], sizeof(val[i]), 1, fin);
	}
	return 0;
}

void free_history(int *r, int *l) {
	free(r);
	free(l);
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
	int hoge;
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
		strcat(buf, "\x20"); //先にバッファにパディングデータを書き込んでおく、segfaultが怖い
		++len_header;
	}

	//メモリ確保、ただしメタデータの大きさを考慮する
	header_2 = (char *)malloc(len_whole);
	if(DEBUG) printf("header size (not including metadata): %ld\n", sizeof(header_2));
	strcpy(header_2, header_1); //デカバッファの1から確保したぴったりサイズの2に移行
	strcat(header_2, buf); //パディングデータをバッファから読み込んで連結
	/*
		下の行のコード、わかりずらく多分忘れるのでメモ
		最後の要素を改行コードに置き換える。
		header_2の最後の要素を示すインデックスはstrlen(header_2) - 1である。
		例: アラインメントで128となっている場合、最後のバイトを示すインデックスは127。
		これをmemmoveを用いて置き換える。memcpyでもいいがmemmoveのほうが安全なのでこっちにしてみる
	*/
	if(DEBUG) printf("last byte: %c", header_2[strlen(header_2) - 1]);
	memmove(&header_2[strlen(header_2) - 1], "\n", sizeof(char)); 


	fwrite(magic_word, sizeof(magic_word), 1, fin);        //1. マジックワード 6byte (char *)
	fwrite(&version_major, sizeof(version_major), 1, fin); //2. メジャーバージョン 1byte (unsigned char)
	fwrite(&version_minor, sizeof(version_minor), 1, fin); //3. マイナーバージョン 1byte (unsigned char)
	fwrite(&len_header, sizeof(len_header), 1, fin);                     //4. ヘッダサイズ 4byte (unsigned short)
	fwrite(header_2, len_header, 1, fin);                         //5. ヘッダ len byte (char *)

	free(header_2);
	return 0;

}

// int main(void) {
// 	FILE *file;
// 	long long i, elements = 1;
// 	long long shape[2] = {10, 10};
// 	int val[10][10] = {
// 		{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
// 		{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
// 		{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
// 		{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
// 		{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
// 		{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
// 		{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
// 		{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
// 		{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
// 		{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
// 	};
// 	int dim = 2;
// 	if((file = fopen("./hogefuga.npy", "wb")) == NULL) {
// 		fprintf(stderr, "おい\n");
// 		return -1;
// 	}

// 	for(i = 0; i < dim; i++) {
// 		elements *= shape[i];
// 	}

// 	int ret = write_npy_header(file, "i4", shape, dim);
// 	fprintf(stderr, "%d\n", ret);

// 	int written = fwrite(val, sizeof(int), elements, file);
// 	fprintf(stderr, "elements: %lld\n", elements);

// 	fseek(file, 0, SEEK_SET);
// 	char nul = '\x0';
// 	for(int unk = 0; unk < 128; unk++) fwrite(&nul, sizeof(char), 1, file);
// 	if(ret != 1) fclose(file);
// 	return 0;
// }



int main(void) {

	//for numpy format
	int debug_counter = 0;
	unsigned short window_size = WINDOW_SIZE;
	long long shape[2] = {0, window_size*2 + 1}; // 左右ウィンドウサイズ + 中心単語分の大きさを持つベクトル
	int dim = 2;
	char dtype[] = "i4"; //integer 8byte = 32bit整数値
	long long id_counter = 0; //0はなにか別のことに使うかもしれないので、++id_counterでインクリメントして0は残しておく

	FILE *vcb_fp, *src_fp, *fid;
	char filename[128], file_head[] = "wiki-cleaned.nostopword.100000.train";
	long long counter, flag, i, j, k;
	int centre_id, context_id, byte_counter = 0;
	unsigned int file_counter = 0;
	HASHREC **vocab_hash = inithashtable(), *htmp, *hprv;
	char ch[MAX_STRING_LENGTH];
	int *history_left = (int *)calloc(window_size, sizeof(int));
	int *history_right = (int *)calloc(window_size, sizeof(int));

	if((vcb_fp = fopen(VCB_FILE_PATH, "r")) == NULL) {
		fprintf(stderr, "Vocabulary File %s not found.\n", VCB_FILE_PATH);
		return 1;
	}

	if((src_fp = fopen(CORPUS_FILE_PATH, "r")) == NULL) {
		fprintf(stderr, "Corpus File %s not found.\n", CORPUS_FILE_PATH);
		return 1;
	}

	while(fscanf(vcb_fp, "%s", ch) != EOF)  {
		hashinsert(vocab_hash, ch, ++id_counter);
	}

	fprintf(stderr, "Hash table generated.\n");

	counter = 0;
	sprintf(filename, "%s%s.%04d", TARGET_DIRECTORY_PATH, file_head, file_counter);
	if(DEBUG) fprintf(stderr, "filename: %s\n", filename);
	fid = fopen(filename, "wb");

	if(fid == NULL) {
		fprintf(stderr, "File %s can't be opened\n", filename);
		return 2;
	}

	if(DEBUG) fprintf(stderr, "DEBUG COUNTER: %d\n", debug_counter++);

	if(write_header_reserve(fid, 128) != 128) {
		fprintf(stderr, "Something went wrong\n");
		return 3;
	}

	if(DEBUG) fprintf(stderr, "DEBUG COUNTER: %d\n", debug_counter++);

	while(1) {

		flag = get_word(ch, src_fp);

		if(flag == 1) {
			if(feof(src_fp)) break;
			else {
				//改行を検知 コンテキストウィンドウはリセット
				for(k = 0; k < window_size; k++) {
					history_left[k] = 0; //コンテキストウィンドウ内の-1は何も意味しない
					//history_right[k] = -1; //どうせこの後右側ウィンドウをロードする
				}
				j = 0; 
				continue;
			}
		}

		htmp = hashsearch(vocab_hash, ch);

		if(htmp == NULL) continue;

		centre_id = htmp->num; // word id

		//history_left[j % window_size] = context_id;

		for(k = 0; k < window_size; k++) {
			if(get_word(ch, src_fp) == 1)  {
				byte_counter += 1;
				break;
			}

			htmp = hashsearch(vocab_hash, ch);
			byte_counter += (strlen(ch) + 1) * sizeof(char); //文字数+空白分 のバイト数
			if(htmp != NULL) history_right[k % window_size] = htmp->num;
		}

		fseek(src_fp, -byte_counter, SEEK_CUR); //先読みした分だけカーソルを戻す
		byte_counter = 0;
		// if(j == 0) {
		// 	//新しい行であるので、右側のコンテキストをロードする
		// 	for(i = 1; i <= window_size; ++i) {
		// 		if(get_word(ch, fid) == 1) {
		// 			if(feof(fid)) {
		// 				//ファイルの終わり
		// 				break;
		// 			}
		// 			//改行らしい
		// 			j == 0; continue;
		// 		}
		// 		htmp = hashsearch(vocab_hash, ch);
		// 		if(htmp == NULL) {
		// 			fprintf(stderr, "Word %s is not in vocab list.\n", ch);
		// 			continue;
		// 		}
				
		// 		history_right[i % window_size] = htmp->num;

		// 	}
		//}


		//left side of centre word
		// for(k = j - 1;  k >= ((j > window_size) ? j - window_size : 0);k--) {
		// 	fprintf(fid, "%lld ", history_left[k % window_size]);
		// }

		//right side of centre word
		// for(k = j + 1; k <= j + window_size; k++) {
		// 	fprintf(fid, "%lld ", history_right[k % window_size]);
		// }
		
		//次の単語を読み込み、history_rightにロードする

		fwrite(&centre_id, sizeof(int), 1, fid);
		fwrite(history_left, sizeof(int), WINDOW_SIZE, fid);
		fwrite(history_right, sizeof(int), WINDOW_SIZE, fid);

		history_left[j % window_size] = centre_id; //左ウィンドウに今回の中心単語をいれる

		j++;
		counter++;
		if((counter % 100000) == 0) fprintf(stderr, "\r\033[0KProcessed %lld token.", counter);
		if((counter % 16777216) == 0) {
			//学習データは16777216(=2^24)個ごとにファイル分割を行いたい
			shape[0] = 16777216;
			fseek(fid, 0, SEEK_SET);
			write_npy_header(fid, dtype, shape, dim);
			fflush(fid);
			fclose(fid);
			file_counter++;
			sprintf(filename, "%s%s.%04d", TARGET_DIRECTORY_PATH, file_head, file_counter);
			fid = fopen(filename, "wb");
			write_header_reserve(fid, 128);
		}
	}

	shape[0] = counter % 16777216;
	fseek(fid, 0, SEEK_SET);
	write_npy_header(fid, dtype, shape, dim);
	fflush(fid);
	fclose(fid);

	fclose(vcb_fp);
	free_table(vocab_hash);
	free_history(history_left, history_right);
}
