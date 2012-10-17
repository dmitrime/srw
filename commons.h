#ifndef COMMONS_H
#define COMMONS_H

#include <cstdio>
#include <vector>

using namespace std;

const unsigned OCT1_2010 = 1285934913;
const unsigned TIMEPOINT = OCT1_2010;

inline double dot_product(vector<double>& a, vector<double>& b)
{
    double sum = 0.0;
    for (unsigned i = 0; i < a.size(); i++)
        sum += a[i]*b[i];
    return sum;
}

inline double logistic_deriv(double x)
{
    if (x > -100 && x < 100)
    {
        double ex = exp(x);
        return ex/((1.0+ex)*(1.0+ex));
    }
    else return 0.0;
}

inline double logistic(double x)
{
    if (x > 30) return 1.0;
    if (x < -30) return 0.0;
    return 1.0 / (1.0 + exp(-x));
}

// Taken from http://www.kaggle.com/c/socialNetwork/forums/t/247/auc-calculation-check/1557#post1557
struct PredictionAnswer 
{
    double prediction;
    unsigned char answer; //this is either 0 or 1
};
//On input, p[] should be in ascending order by prediction
double calculateAUC(vector<PredictionAnswer> &p)
{
    unsigned int i,truePos,tp0,accum,tn,ones=0;
    float threshold; //predictions <= threshold are classified as zeros
    unsigned count = p.size();
    for (i=0;i<count;i++)
        ones+=p[i].answer;
    if (0==ones || count==ones)
        return 1;

    truePos=tp0=ones;
    accum=tn=0;
    threshold=p[0].prediction;

    for (i=0;i<count;i++) 
    {
        if (p[i].prediction!=threshold) 
        { //threshold changes
            threshold=p[i].prediction;
            accum+=tn*(truePos+tp0); //2* the area of trapezoid
            tp0=truePos;
            tn=0;
        }
        tn+= 1- p[i].answer; //x-distance between adjacent points
        truePos-= p[i].answer;            
    }
    accum+=tn*(truePos+tp0); //2* the area of trapezoid
    return (double)accum/(2*ones*(count-ones));
}

void load_ids(const char *users_ids, vector<user_id>& users)
{
    FILE *inp = fopen64(users_ids, "r");
    user_id ind = 0;
    while(fscanf(inp, "%u", &ind) == 1)
        users.push_back(ind);
    fclose(inp);
}

#endif

