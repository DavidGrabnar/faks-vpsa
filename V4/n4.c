#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define DEFAULT_N 1000000;
#define DEFAULT_T 1;

int getDivisorSum(int n);

// command: ./n4 <n> <t>
// n - range [1, n] will be used for processing
// t - number of threads
int main(int argc, char **argv) {
    int n = DEFAULT_N;
    int t = DEFAULT_T;
    if (argc >= 2) {
        n = atoi(argv[1]);
    }
    if (argc >= 3) {
        t = atoi(argv[2]);
    }


    // since we skip 0, we shift index access by -1
    int* numbers = calloc(n, sizeof(int));
    long sum = 0;
    int divisorSum;
    double start, half, end;

    omp_set_num_threads(t);
    #pragma omp parallel
    {
        start = omp_get_wtime();
        #pragma omp for schedule(dynamic, 1000)
        for (int i = 1; i <= n; i++) {
            numbers[i - 1] = getDivisorSum(i);
        }

        half = omp_get_wtime();

        #pragma omp for reduction(+: sum) private(divisorSum)
        for (int i = 1; i <= n; i++) {
            divisorSum = numbers[i - 1];

            // ensure its in array AND it didn't already occur and are not the same AND they are ambicable pair
            if (divisorSum <= n && divisorSum > i && numbers[divisorSum - 1] == i) {
                sum += i + divisorSum;
            }
        }
        end = omp_get_wtime();
    }

    printf("Done.\nSum: %ld.\nGenerating divisor sums took %f seconds.\nGenerating final sum took %f seconds.\nProgram took %f seconds.\n", sum, half - start, end - half, end - start);

    free(numbers);
    return 0;
}

int getDivisorSum(int n) {
    int sum = 1;
    for (int i = 2; i <= sqrt(n); i++) {
        if (n % i == 0) {
            sum += i;

            if (i != n / i) {
                sum += n / i;
            }
        }
    }

    return sum;
}

/* 

Meritve:

N = 1_000_000
Čas ~ povprečje 5 meritev

#niti (T) | Čas (t) [s] | Pohitritev (S)
-----------------------------------------
    1     |    5,95     |       /       
    2     |    2,99     |     1,99
    4     |    1,50     |     3,96
    8     |    0,76     |     7,83
    16    |    0,38     |    15,66
    32    |    0,19     |    31,32

ts ~ sekvenčno, T = 1
tp ~ paralelno, T > 1

Pristopi delitve dela:
static, dynamic, guided

Izbral sem dynamic delitev, ker se generiranje vsote deliteljev razlikuje v zahtevnosti
(večja številka, bolj zahtevna). 
Static delitev bi izbral v primeru, da je procesiranje za vsa števila prilbližno enako zahtevno.

*/
