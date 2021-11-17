#include <time.h>
#include <stdlib.h>
#include <stdio.h>

int* generate(int n);
void print(int* arr, int n);
void sort(int* arr, int n);
void swap(int* arr, int i, int j);

int main(int argc, char **argv)
{
    srand(time(NULL));

    int t = 1;
    int n = 8;
    if (argc > 1) {
        t = atoi(argv[1]);
    }
    if (argc > 2) {
        n = atoi(argv[2]);
    }

    int* arr = generate(n);
    // int arr[] = {2, 1, 4, 9, 5, 3, 6, 10};

    sort(arr, n);
    print(arr, n);
    return 0;
}

int* generate(int n) {
    int* arr = (int*) calloc(n, sizeof(int));
    for (int i = 0; i < n; i++)
    {
        arr[i] = (int) rand() % (2 * n);
    }
    return arr;
}

void print(int* arr, int n) {
    for (int i = 0; i < n; i++) {
        printf("%d ,", arr[i]);
    }
    printf("\n");
}

void sort(int* arr, int n) {
    int sorted = 0;
    while(!sorted) {
        sorted = 1;
        for (int i = 1; i < n - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                swap(arr, i, i + 1);
                sorted = 0;
            }
        }
        for (int i = 0; i < n - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                swap(arr, i, i + 1);
                sorted = 0;
            }
        }
    }
}

void swap(int* arr, int i, int j) {
    int tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
}
