#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <stdlib.h>
#include <openssl/rand.h>
#include <string.h>
#include <stdbool.h>
#include <stdarg.h>

//compilation gcc main.c -o as2.exe -IC:/msys64/mingw64/include -LC:/msys64/mingw64/lib -lcrypto -lssl

typedef struct{
    uint64_t size;
    double* samples;
}stat_sample_t;

typedef struct{
    double upper_limit;
    double lower_limit;
    double range;
}confidence_interval_t;

//RNG`s
double get_rand_01();
double* get_rands(uint64_t size);

//General sample operations
void Init_Sample(stat_sample_t* s, uint64_t size); 
void Clear_Samples(stat_sample_t* first, ...);
void Sample_Analyze(stat_sample_t* s, double* mean, double* var, bool bias, double* st_dev);

stat_sample_t* Get_Sample_exp(uint64_t size, double lambda);
stat_sample_t* Copy_Sample(stat_sample_t* s1);
stat_sample_t* Get_Sample_u(const stat_sample_t* base, double lambda);


int main(){
    uint64_t rands_size = 100;
    double* first_smp = get_rands(rands_size);

    for(uint64_t i = 0; i < rands_size; i++){
        printf("r[%3d] = %.3lf  ", i+1, first_smp[i]);
        if((i + 1) % 10 == 0) printf("\n");
    }

    printf("\n");

    stat_sample_t* sample = Get_Sample_exp(100, (double)4);

    for(uint64_t i = 0; i < 100; i++){
        printf("e[%3d] = %.3lf  ", i+1, sample->samples[i]);
        if((i + 1) % 10 == 0) printf("\n");
    }

    Clear_Samples(sample, NULL);

    return 0;
}

void Clear_Samples(stat_sample_t* first, ...){
    if(first == NULL) return;

    va_list args;
    va_start(args, first);

    stat_sample_t* stream = first;

    while(stream != NULL){
        if(stream->samples != NULL){
            free(stream->samples);
            stream->samples = NULL;
        }
        free(stream);

        stream = va_arg(args, stat_sample_t*);
    }

    va_end(args);
}

void Init_Sample(stat_sample_t* s, uint64_t size){
    s->size = size;
    s->samples = calloc(s->size,sizeof(double));
}

stat_sample_t* Copy_Sample(stat_sample_t* s1){

    if(s1 == NULL) return NULL;

    stat_sample_t* s2 = malloc(sizeof(stat_sample_t));
    if(s2 == NULL) return NULL;
    
    Init_Sample(s2, s1->size);
    if(s2->samples == NULL){
        free(s2);
        return NULL;
    }

    memcpy(s2 -> samples, s1 -> samples, s1->size * sizeof(double));

    return s2;
}

double* get_rands(uint64_t size){

    if(size > 1e+10 || size == 0){
        printf("\nInsufficient size given, generation impossible, check the parameters");
        return NULL;
    }

    double* rands = malloc(size * sizeof(double));
    uint64_t* raw = malloc(size * sizeof(uint64_t));

    if(rands == NULL || raw == NULL){
        printf("\nMemory error, rands not generated");
        free(rands);
        free(raw);
        return NULL;
    }

    RAND_bytes((unsigned char*)raw, size*sizeof(uint64_t));

    for(uint64_t i = 0; i < size ; i++){
        rands[i] = (double)raw[i] / (double)UINT64_MAX;
    }

    free(raw);

    return rands;
}

stat_sample_t* Get_Sample_exp(uint64_t size, double lambda){
    stat_sample_t* sample = malloc(sizeof(stat_sample_t));

    sample->size = size;
    sample->samples = get_rands(size);

    double stream = 0;
    for(uint64_t i = 0; i < size; i++){
        
        stream = sample->samples[i];
        (stream < 1e-15) ? stream += 1e-15 : 0;
        sample->samples[i] = (-(double)1 / lambda) * log(stream);
    }
    return sample;
}

stat_sample_t* Get_Sample_u(const stat_sample_t* base, double lambda){
    if(base == NULL || base->samples == NULL) return 0;

    stat_sample_t* y = malloc(sizeof(stat_sample_t));
    Init_Sample(y, base->size);

    for(uint64_t i = 0; i < base->size; i++){
        y->samples[i] = (double)1 - exp(-lambda * base->samples[i]);
    }

    return y;
}

void Sample_Analyze(stat_sample_t* s, double* mean, double* var, bool bias, double* st_dev){

    if(s == NULL) return;
    if(s->size < 2 || s->size == 0) return;

    double s_mean = 0;
    double s_var = 0;
    uint64_t q_bias = (bias) ? 1 : 0;

    for(uint64_t i = 0; i < s->size ; i++){
        s_mean += s->samples[i];
    }

    s_mean /= (double)s->size;

    for(uint64_t i = 0; i < s->size; i++){
        double diff = s->samples[i] - s_mean;
        s_var += diff * diff;
    }

    s_var /= (double)(s->size - q_bias);
    double s_st_dev = sqrt(s_var);

    *mean = s_mean;
    *var = s_var;
    *st_dev = s_st_dev;
}


//NULL