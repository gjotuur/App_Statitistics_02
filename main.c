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

typedef struct{
    double d_n;
    double k_n;
    double z_gamma;
    bool criteria_passed;
}hypothesis_result_t;

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

//Statistical functions
double F_Kolmogorov(double z, double epsilon);
double Find_Z_gamma(double gamma, double epsilon);


int main(){
    stat_sample_t* sample = Get_Sample_exp(100, 1);
    stat_sample_t* sample2 = Get_Sample_exp(100, 1.2);

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

double F_Kolmogorov(double z, double epsilon){
    if(z <= 0.0) return (double)0;
    if(z <= 0.15) return (double)0;

    double s = (double)0;
    int k = 1;

    while(true){
        double  sign = (k%2 == 0) ? 1.0 : -1.0,
                term = sign * exp(-2.0 * k * k * z * z);
        s += term;

        if(fabs(term) < epsilon){
            break;
        }

        k++;
        if(k > 1000) break;
    }
    double res = 1.0 + 2.0 * s;

    if(res < 0.0) return 0.0;
    if(res > 1.0) return 1.0;

    return res;
}

double Find_Z_gamma(double gamma, double epsilon){
    double  target = 1.0 - gamma, low = 0.0,
            high = 5.0, mid = 0.0;

    for(int i = 0; i < 100; i++){
        mid = low + ((high - low) / 2.0);
        double value = F_Kolmogorov(mid, 1e-6);

        if(value < target){
            low = mid;
        } else {
            high = mid;
        }
    }
    return mid;
}

//Utility function for qsort
int cmp_double(const void* a, const void* b){
    double da = *(const double*)a;
    double db = *(const double*)b;

    return (da > db) - (da < db);
}

hypothesis_result_t Kolmogorov_Criteria(stat_sample_t* y, double gamma, double epsilon){
    hypothesis_result_t res;
    uint64_t N = y->size;

    qsort(y->samples, N, sizeof(double), cmp_double);

    double max_d = 0.0;
    
    for(uint64_t i = 1; i <= N; i++){
        double d_plus = ((double)i / N) - y->samples[i - 1];
        double d_minus = y->samples[i - 1] - ((double)(i-1) / N);

        if(d_plus > max_d) max_d = d_plus;
        if(d_minus > max_d) max_d = d_plus;
    }
    res.d_n = max_d;

    res.k_n = sqrt((double)N) * res.d_n;
    res.z_gamma = Find_Z_gamma(gamma, epsilon);
    res.criteria_passed = (res.k_n < res.z_gamma);

    return res;
}

//NULL