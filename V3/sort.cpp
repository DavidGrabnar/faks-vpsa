#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>

int* generate(int n);
void print(int* arr, int n);
void sort(int* arr, int n, int t);
void swap(int* arr, int i, int j);
int step(int* arr, int n, int t, int start);
void* evaluate(void* arg);
int min(int a, int b);

struct params {
    int offset;
    int count;
    int n;
    int* arr;
};

pthread_barrier_t barrier;

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
    for (int i = 0; i < t; i++) {
        int ti = (i + 1) * n / t - i * n / t;
        printf("at %d = %d\n", i, ti);
    }
    printf("Starting sort\n");

    // print(arr, n);
    time_t start = time(NULL);
    sort(arr, n, t);
    time_t end = time(NULL);
    // print(arr, n);
    printf("Sort took %ld seconds.\n", end - start);
    return 0;
}

void* evaluate(void* arg) {
    struct params p = *((struct params*) arg);

    int *changed = (int *)malloc(sizeof(int));
    *changed = 0;
    for (int i = p.offset; i < min(p.offset + p.count + p.count % 2, p.n - 1); i += 2) {
        // printf("Comparing %d %d : %d < %d\n", i, i + 1, p.arr[i], p.arr[i + 1]);
        if (p.arr[i] > p.arr[i + 1]) {
            swap(p.arr, i, i + 1);
            *changed = 1;
        }
    }
    pthread_barrier_wait(&barrier);
    pthread_exit(changed);
    return nullptr;
}

int* generate(int n) {
    int* arr = (int*) calloc(n, sizeof(int));
    for (int i = 0; i < n; i++) {
        arr[i] = (int) rand() % (2 * n);
    }
    return arr;
}

void print(int* arr, int n) {
    for (int i = 0; i < n; i++) {
        printf("%d, ", arr[i]);
    }
    printf("\n");
}

void sort(int* arr, int n, int t) {
    int sorted = 0;
    while(!sorted) {
        sorted = 1 - (step(arr, n, t, 0) | step(arr, n, t, 1));
    }
}

int step(int* arr, int n, int t, int start) {
    int changed = 0;
    int offset = start;

    pthread_barrier_init(&barrier, NULL, t);

    pthread_t threads[t];
    for (int i = 0; i < t; i++) {
        int ti = (i + 1) * n / t - i * n / t;
        struct params* p = (struct params *)malloc(sizeof(struct params));
        p->offset = offset;
        p->count = ti;
        p->n = n;
        p->arr = arr;

        pthread_create(&threads[i], NULL, evaluate, p);
        
        // multiple loops per step
        // for (int i = p.offset; i < min(p.offset + p.count + ti % 2, n - 1); i += 2) {
        //     printf("Comparing %d %d : %d < %d\n", i, i + 1, arr[i], arr[i + 1]);
        //     if (arr[i] > arr[i + 1]) {
        //         swap(arr, i, i + 1);
        //         changed = 1;
        //     }
        // }
        offset += ti;
    }

    // multithreaded step
    for (int i = 0; i < t; i++) {
        void *result;
        pthread_join(threads[i], &result);
        int threadChanged = *((int*) result);
        if (threadChanged) {
            changed = 1;
        }
    }

    // single loop per step
    // for (int i = start; i < n - 1; i += 2) {
    //     if (arr[i] > arr[i + 1]) {
    //         swap(arr, i, i + 1);
    //         changed = 1;
    //     }
    // }
    return changed;
}

void swap(int* arr, int i, int j) {
    int tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
}

int min(int a, int b) {
    return a < b ? a : b;
}

/*
STATS:

N = 1_000_000

Threads[T] | Duration[s] | Speedup[-]
-------------------------------------
     1     |     48      |
     2     |     29      |    1.65


*/