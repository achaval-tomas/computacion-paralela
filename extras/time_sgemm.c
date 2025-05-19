// Basado en:
// https://gist.githubusercontent.com/xianyi/5780018/raw/442c007cf29c3387ee8f98d853da97408ab204a4/time_dgemm.c
//
// gcc -o time_sgemm time_sgemm.c -lopenblas
#include "stdio.h"
#include "stdlib.h"
#include "sys/time.h"
#include "time.h"

extern void sgemm_(char*, char*, int*, int*,int*, float*, float*, int*, float*, int*, float*, float*, int*);

int main(int argc, char* argv[])
{
  int i;
  printf("test!\n");
  if(argc<4){
    printf("Input Error\n");
    return 1;
  }

  int m = atoi(argv[1]);
  int n = atoi(argv[2]);
  int k = atoi(argv[3]);
  int sizeofa = m * k;
  int sizeofb = k * n;
  int sizeofc = m * n;
  char ta = 'N';
  char tb = 'N';
  float alpha = 1.2;
  float beta = 0.001;

  struct timeval start,finish;
  double duration;

  float* A = (float*)malloc(sizeof(float) * sizeofa);
  float* B = (float*)malloc(sizeof(float) * sizeofb);
  float* C = (float*)malloc(sizeof(float) * sizeofc);

  srand((unsigned)time(NULL));

  for (i=0; i<sizeofa; i++)
    A[i] = i%3+1;//(rand()%100)/10.0;

  for (i=0; i<sizeofb; i++)
    B[i] = i%3+1;//(rand()%100)/10.0;

  for (i=0; i<sizeofc; i++)
    C[i] = i%3+1;//(rand()%100)/10.0;
  //#if 0
  printf("m=%d,n=%d,k=%d,alpha=%lf,beta=%lf,sizeofc=%d\n",m,n,k,alpha,beta,sizeofc);
  gettimeofday(&start, NULL);
  sgemm_(&ta, &tb, &m, &n, &k, &alpha, A, &m, B, &k, &beta, C, &m);
  gettimeofday(&finish, NULL);

  duration = ((double)(finish.tv_sec-start.tv_sec)*1000000 + (double)(finish.tv_usec-start.tv_usec)) / 1000000;
  double gflops = 2.0 * m *n*k;
  gflops = gflops/duration*1.0e-6;

  FILE *fp;
  fp = fopen("timeSGEMM.txt", "a");
  fprintf(fp, "%dx%dx%d\t%lf s\t%lf MFLOPS\n", m, n, k, duration, gflops);
  fclose(fp);

  free(A);
  free(B);
  free(C);
  return 0;
}
