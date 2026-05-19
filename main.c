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

typedef enum{
    Kolmogorov = 0,
    Pearson = 1,
    Empty_Boxes = 2,
    Smirnov = 3
} hypothesis_tests_t;

typedef struct{
    uint64_t n;
    hypothesis_tests_t method;
    double real_lambda;
    double test_lambda;
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
double F_Khi2(double x, int k);
double Find_Khi2_gamma(double gamma, int k, double epsilon);

//Hypothesis tests
bool Kolmogorov_criteria(const stat_sample_t* raw_sample, double hypothetical_lambda, double gamma, double epsilon);
bool Pearson_criteria(const stat_sample_t* raw_sample, double hypothetical_lambda, double gamma, double epsilon);
void Run_Tests(double test_lambda, double real_lambda1, double real_lambda2, double gamma, hypothesis_tests_t chosen_type);

int main(){
    Run_Tests(1.0, 1.0, 2.0, 0.05, Kolmogorov);
    Run_Tests(1.0, 1.0, 2.0, 0.05, Pearson);
    Run_Tests(1.0, 1.0, 2.0, 0.05, Empty_Boxes);
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
    if(z <= 1e-7) return (double)0;

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

double F_Khi2(double x, int k){
    if (x <= 0.0) return 0.0;

    double a = (double)k / 2.0;
    double log_term = -x/2.0 + a * log(x/2.0) - lgamma(a + 1.0);
    double term = exp(log_term);
    double sum = term;

    for (int n = 1; n < 10000; n++) {
        term *= (x / 2.0) / (a + (double)n);
        sum += term;

        if (fabs(term) < 1e-10 * sum) break;
    }

    return (sum > 1.0) ? 1.0 : sum;
}

double Find_Khi2_gamma(double gamma, int k, double epsilon) {
    double target = 1.0 - gamma;
    double low = 0.0, mid = 0.0;
    double high = (double)k + 10.0 * sqrt((double)k);

    for (int i = 0; i < 200; i++) {
        mid = low + (high - low) / 2.0;
        double value = F_Khi2(mid, k);

        if (value < target) {
            low = mid;
        } else {
            high = mid;
        }

        if ((high - low) < epsilon) break;
    }

    return mid;
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

double F_Phi(double z) {
    return 0.5 * (1.0 + erf(z / sqrt(2.0)));
}

double Find_Z_normal(double gamma, double epsilon){
    double target = 1.0 - gamma;
    double low = -10.0, high = 10.0, mid = 0.0;

    for (int i = 0; i < 200; i++) {
        mid = (low + high) / 2.0;
        if (F_Phi(mid) < target) low = mid;
        else high = mid;
        if ((high - low) < epsilon) break;
    }
    return mid;
}

//Utility function for qsort
int cmp_double(const void* a, const void* b){
    double da = *(const double*)a;
    double db = *(const double*)b;

    return (da > db) - (da < db);
}

bool Kolmogorov_criteria(const stat_sample_t* raw_sample, double hypothetical_lambda, double gamma, double epsilon) {
    
    uint64_t N = raw_sample->size;

    stat_sample_t* Y = Get_Sample_u(raw_sample, hypothetical_lambda);
    
    qsort(Y->samples, N, sizeof(double), cmp_double);

    double max_diff = 0.0;
    for (uint64_t i = 1; i <= N; i++) {
        double d_plus = ((double)i / N) - Y->samples[i - 1];
        double d_minus = Y->samples[i - 1] - ((double)(i - 1) / N);
        
        if (d_plus > max_diff)  max_diff = d_plus;
        if (d_minus > max_diff) max_diff = d_minus;
    }

    double k_n = sqrt((double)N) * max_diff;
    double z_gamma = Find_Z_gamma(gamma, epsilon);

    return (k_n < z_gamma);
}

bool Pearson_criteria(const stat_sample_t* raw_sample, double hypothetical_lambda, double gamma, double epsilon) {
    uint64_t N = raw_sample->size;

    int m = (int)(30.0 * (double)N / 1000.0);
    if (m < 2) m = 2;

    int k = m - 1;

    double p_theory = 1.0 / (double)m;

    stat_sample_t* Y = Get_Sample_u(raw_sample, hypothetical_lambda);
    if (Y == NULL) return false;

    int* freq = calloc(m, sizeof(int));
    if (freq == NULL) { Clear_Samples(Y, NULL); return false; }

    for (uint64_t i = 0; i < N; i++) {
        int bin = (int)(Y->samples[i] * m);
        if (bin >= m) bin = m - 1;
        freq[bin]++;
    }

    double chi2 = 0.0;
    double expected = (double)N * p_theory;

    for (int i = 0; i < m; i++) {
        double diff = (double)freq[i] - expected;
        chi2 += (diff * diff) / expected;
    }

    double chi2_gamma = Find_Khi2_gamma(gamma, k, epsilon);

    free(freq);
    Clear_Samples(Y, NULL);

    return (chi2 < chi2_gamma);
}

bool EmptyBoxes_criteria(const stat_sample_t* raw_sample, double hypothetical_lambda, double gamma, double epsilon) {
    uint64_t n = raw_sample->size;

    int r = (int)(n / 2);
    if (r < 2) r = 2;

    double rho = (double)n / (double)r;

    stat_sample_t* Y = Get_Sample_u(raw_sample, hypothetical_lambda);
    if (Y == NULL) return false;

    int* freq = calloc(r, sizeof(int));
    if (freq == NULL) { Clear_Samples(Y, NULL); return false; }

    for (uint64_t i = 0; i < n; i++) {
        int bin = (int)(Y->samples[i] * r);
        if (bin >= r) bin = r - 1;
        freq[bin]++;
    }

    int mu0 = 0;
    for (int i = 0; i < r; i++) {
        if (freq[i] == 0) mu0++;
    }

    double exp_neg_rho = exp(-rho);
    double m_f   = exp_neg_rho;
    double var_f  = exp_neg_rho * (1.0 - (1.0 + rho) * exp_neg_rho);

    double threshold = (double)r * m_f + Find_Z_normal(gamma, epsilon) * sqrt((double)r * var_f);

    free(freq);
    Clear_Samples(Y, NULL);

    return ((double)mu0 <= threshold);
}

const char* Method_Name(hypothesis_tests_t method){
    switch(method){
        case Kolmogorov: return "Kolmogorov";
        case Pearson: return "Pearson";
        case Empty_Boxes: return "Empty Boxes";
        case Smirnov: return "Smirnov";
    }
    return "Unknown";
}

void Run_Tests(double test_lambda, double real_lambda1, double real_lambda2, double gamma, hypothesis_tests_t chosen_type){
    
    uint64_t tests = 4, sample_sizes[] = {100,1000,10000,100000};
    
    hypothesis_result_t results_t1[tests * 2];
    
    for(uint64_t i = 0; i < tests; i++){
        double first_check = real_lambda1, second_check = real_lambda2;

        stat_sample_t* test_sample_1 = Get_Sample_exp(sample_sizes[i], first_check);
        stat_sample_t* test_sample_2 = Get_Sample_exp(sample_sizes[i], second_check);

        results_t1[2*i].n = sample_sizes[i];
        results_t1[2*i + 1].n = sample_sizes[i];

        results_t1[2*i].real_lambda = first_check;
        results_t1[2*i+1].real_lambda = second_check;

        results_t1[2*i].test_lambda = results_t1[2*i+1].test_lambda = 1.0;

        double hypothetical_lambda = 1.0;
        switch(chosen_type){
            case Kolmogorov:
                results_t1[2*i].criteria_passed = Kolmogorov_criteria(test_sample_1, hypothetical_lambda, gamma, 1e-7);
                results_t1[2*i + 1].criteria_passed = Kolmogorov_criteria(test_sample_2, hypothetical_lambda, gamma, 1e-7);
                results_t1[2 * i].method = Kolmogorov;
                results_t1[2 * i + 1].method = results_t1[2 * i].method;
                break;
            case Pearson:
                results_t1[2*i].criteria_passed = Pearson_criteria(test_sample_1, hypothetical_lambda, gamma, 1e-7);
                results_t1[2*i + 1].criteria_passed = Pearson_criteria(test_sample_2, hypothetical_lambda, gamma, 1e-7);
                results_t1[2 * i].method = Pearson;
                results_t1[2 * i + 1].method = results_t1[2 * i].method;
                break;
            case Empty_Boxes:
                results_t1[2*i].criteria_passed = EmptyBoxes_criteria(test_sample_1, hypothetical_lambda, gamma, 1e-7);
                results_t1[2*i + 1].criteria_passed = EmptyBoxes_criteria(test_sample_2, hypothetical_lambda, gamma, 1e-7);
                results_t1[2 * i].method = Empty_Boxes;
                results_t1[2 * i + 1].method = results_t1[2 * i].method;
                break;
        }
        Clear_Samples(test_sample_1, test_sample_2, NULL);
    }

    for(uint64_t i = 0; i < tests; i++){
        printf("Test #%d\n", i + 1);
        printf("Case%d: Method [%s], Real parameter = %.2lf, checking parameter %.2lf, sample size %llu, test - %s", 
            (i % 2 == 0 ? 1 : 2), Method_Name(results_t1[2*i].method), results_t1[2*i].real_lambda, results_t1[2*i].test_lambda, results_t1[2*i].n, results_t1[2*i].criteria_passed ? "Passed" : "Not passed");
        printf("\nCase%d: Method [%s], Real parameter = %.2lf, checking parameter %.2lf, sample size %llu, test - %s", 
            (i % 2 == 0 ? 1 : 2), Method_Name(results_t1[2*i + 1].method),results_t1[2*i + 1].real_lambda, results_t1[2*i + 1].test_lambda, results_t1[2*i + 1].n, results_t1[2*i + 1].criteria_passed ? "Passed" : "Not passed");
        printf("\n");
    }
}

//NULL