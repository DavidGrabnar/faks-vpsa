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
void* evaluateControl(void* arg);
int min(int a, int b);

struct params {
    int offset;
    int count;
    int n;
    int index;
    int* arr;
};

pthread_barrier_t barrierOdd;
pthread_barrier_t barrierEven;
pthread_barrier_t barrierProceed;
int sorted = 0;
int* changed;

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
    }
    printf("Starting sort\n");

    time_t start = time(NULL);
    sort(arr, n, t);
    time_t end = time(NULL);

    printf("Sort took %ld seconds.\n", end - start);
    return 0;
}

void* evaluate(void* arg) {
    struct params p = *((struct params*) arg);

    while(!sorted) {
        int threadChanged = step(p.arr, p.n, p.offset, p.count);

        pthread_barrier_wait(&barrierOdd);

        threadChanged |= step(p.arr, p.n, p.offset + 1, p.count);

        if (threadChanged) {
            changed[p.index] = 1;
        }
        pthread_barrier_wait(&barrierEven);

        pthread_barrier_wait(&barrierProceed);
    }

    return nullptr;
}

void* evaluateControl(void* arg) {
    int t = *((int*) arg);

    while(!sorted) {
        pthread_barrier_wait(&barrierOdd);

        pthread_barrier_wait(&barrierEven);

        int anyChanged = 0;
        for (int i = 0; i < t; i++) {
            if (changed[i]) {
                anyChanged = 1;
                changed[i] = 0;
            }
        }
        if (!anyChanged) {
            sorted = 1;
        }

        pthread_barrier_wait(&barrierProceed);
        
    }

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
    pthread_t threads[t];
    pthread_t controlThread;

    pthread_barrier_init(&barrierOdd, NULL, t + 1);
    pthread_barrier_init(&barrierEven, NULL, t + 1);
    pthread_barrier_init(&barrierProceed, NULL, t + 1);

    changed = (int *)calloc(t, sizeof(int));
    for(int i = 0; i < t; i++) {
        changed[i] = 0;
    }

    int offset = 0;
    for (int i = 0; i < t; i++) {
        int ti = (i + 1) * n / t - i * n / t;
        struct params* p = (struct params *)malloc(sizeof(struct params));
        p->offset = offset;
        p->count = ti;
        p->n = n;
        p->index = i;
        p->arr = arr;

        pthread_create(&threads[i], NULL, evaluate, p);
        
        offset += ti;
    }
    pthread_create(&controlThread, NULL, evaluateControl, &t);

    for (int i = 0; i < t; i++) {
        pthread_join(threads[i], NULL);
    }
    pthread_join(controlThread, NULL);
}

int step(int* arr, int n, int offset, int count) {
    int changed = 0;
    for (int i = offset; i < min(offset + count + count % 2, n - 1); i += 2) {
        if (arr[i] > arr[i + 1]) {
            swap(arr, i, i + 1);
            changed = 1;
        }
    }
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

Optimisation with reduces phase count (iterates until phase has no changes)
Complied with -O2 flag
Ran on PC
N = 100_000

Threads[T] | Duration[s] | Speedup[-]
-------------------------------------
     1     |     12      |
     2     |     10      |    1.2   
     4     |     12      |    1
     8     |     18      |    0.66
     16    |     32      |    0.38


*/