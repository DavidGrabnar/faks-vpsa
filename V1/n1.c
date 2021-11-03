#include <time.h>
#include <stdlib.h>
#include <stdio.h>

double *Random(int n);
double **Matrix(double *A, int n, int r, int c);
double *Max(double *A, int n);

int main()
{
    srand(time(NULL));

    int n, r;
    printf("Vnesi n: ");
    scanf("%d", &n);
    printf("Vnesi r: ");
    scanf("%d", &r);

    double *vec = Random(n);
    printf("1D:\n");
    for (int i = 0; i < n; i++)
    {
        printf("%.2f ", vec[i]);
    }
    printf("\n");

    // ceil(A/B) ~ (A + (B - 1)) / B - to avoid floating division and ceil
    int c = (n + r - 1) / r;
    double **mat = Matrix(vec, n, r, c);
    printf("2D:\n");
    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            printf("%.2f ", mat[i][j]);
        }
        printf("\n");
    }

    double *max = Max(vec, n);
    printf("Najvecja vrednost: %.2f na naslovu: %p\n", *max, max);
}

double *Random(int n)
{
    double *randoms = (double *)calloc(n, sizeof(double));
    for (int i = 0; i < n; i++)
    {
        randoms[i] = (double)rand() / RAND_MAX;
    }
    return randoms;
}

double **Matrix(double *A, int n, int r, int c)
{
    double **randoms = (double **)calloc(r, sizeof(double *));
    for (int i = 0; i < r; i++)
    {
        randoms[i] = (double *)calloc(c, sizeof(double));
        for (int j = 0; j < c; j++)
        {
            int idx = i * c + j;
            if (idx >= n)
            {
                break;
            }
            randoms[i][j] = A[idx];
        }
    }
    return randoms;
}

double *Max(double *A, int n)
{
    double *max = A;

    for (int i = 0; i < n; i++)
    {
        if (A[i] > *max)
        {
            max = &A[i];
        }
    }

    return max;
}