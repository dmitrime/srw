#ifndef PARAMS_H
#define PARAMS_H

#include <iostream>
#include <vector>
#include <cmath>
#include <map>

#include "commons.h"

class Params
{
    typedef unordered_map<user_id, vector<double> > derivs;
    typedef unordered_map<user_id, derivs> edge_derivs;

    typedef map<pair<user_id,user_id>, vector<double> > edge_feature_matrix;

    private:
        Graph *graph;
        edge_feature_matrix features;
        edge_derivs dQ;
        vector<double> wvec;

        vector<double> edge_features(user_id from, user_id to, unsigned data, unsigned mutual)
        {
            // These are the four current features + the intercept term (at index 4):
            // Edge creation time since TIMEPOINT using 3 functions  
            // and the number of mutual friends.
            // More features can be added, but the training time will increase.
            vector<double> fvec(wvec.size(), 1.0);
            fvec[0] = pow(fabs(TIMEPOINT-data), -0.1);
            fvec[1] = pow(fabs(TIMEPOINT-data), -0.3);
            fvec[2] = pow(fabs(TIMEPOINT-data), -0.5);
            fvec[3] = mutual;
            return fvec;
        }

        // subtract the mean and divices by the st.dev
        // for all features except the last one which is the intercept
        void standardize(vector<vector<double> > &vals)
        {
            vector<double> mean(vals.size(), 0.0);
            vector<double> std(vals.size(), 0.0);
            for (unsigned i = 0; i < vals.size(); i++)
            {
                for (unsigned j = 0; j < vals[i].size(); j++)
                    mean[i] += vals[i][j];
                mean[i] /= vals[i].size();

                for (unsigned j = 0; j < vals[i].size(); j++)
                {
                    double t = vals[i][j]-mean[i];
                    std[i] += t*t;
                }
                std[i] = sqrt(std[i]/vals[i].size());
            }

            for (edge_feature_matrix::iterator it = features.begin(); it != features.end(); it++)
            {
                vector<double> &fvec = it->second;
                for (unsigned i = 0; i < mean.size(); i++)
                    fvec[i] = (fvec[i]-mean[i]) / std[i];
            }
        }

        void extract_features(unordered_map<user_id, double>& m)
        {
            unsigned nvert = graph->vertex_count;
            vector<vector<double> > vals(wvec.size()-1);
            for (unsigned id = 0; id < nvert; id++)
            {
                for (Graph::iterator e = graph->iterate_outgoing_edges(id); !e.end(); e++)
                {
                    user_id fr = (*e).v2;
                    pair<user_id, user_id> p = make_pair(id, fr);
                    features[p] = edge_features(id, fr, (*e).data, m[fr]);
                    vector<double> &fvec = features[p];

                    for (unsigned i = 0; i < fvec.size()-1; i++)
                        vals[i].push_back(fvec[i]);
                }
            }
            standardize(vals);
        }

        inline vector<double>& edge_vector(user_id a, user_id b)
        {
            return features[make_pair(a,b)];
        }

    public:
        Params(vector<double> &w, Graph *lgraph, unordered_map<user_id, double>& m)
        {
            wvec = w;
            graph = lgraph;
            extract_features(m);
        }
        inline double edge_strength(user_id from, user_id to)
        {
            vector<double>& fvec = edge_vector(from, to);
            return logistic(dot_product(fvec, wvec));
        }
        void recalculate_derivs()
        {
            dQ = edge_derivs();
            unsigned nvert = graph->vertex_count;
            for (unsigned j = 0; j < nvert; j++)
                dQ[j] = derivs();
            #pragma omp parallel for
            for (unsigned j = 0; j < nvert; j++)
            {
                for (unsigned w = 0; w < wvec.size(); w++)
                {
                    double fw_sum = 0.0, dfw_sum = 0.0;
                    for (Graph::iterator e = graph->iterate_outgoing_edges(j); !e.end(); e++)
                    {
                        user_id nei = (*e).v2;
                        fw_sum += edge_strength(j, nei);

                        vector<double>& fvec = edge_vector(j,nei);
                        dfw_sum += logistic_deriv(dot_product(fvec, wvec))*fvec[w];

                        if (w == 0)
                            dQ[j][nei] = vector<double>(wvec.size(), 0.0);
                    }
                    for (Graph::iterator e = graph->iterate_outgoing_edges(j); !e.end(); e++)
                    {
                        user_id nei = (*e).v2;

                        double fw_ju = edge_strength(j, nei);
                        vector<double>& fvec = edge_vector(j,nei);
                        double dfw_ju = logistic_deriv(dot_product(fvec, wvec))*fvec[w];

                        dQ[j][nei][w] = (dfw_ju*fw_sum - fw_ju*dfw_sum) / (fw_sum*fw_sum);
                    }
                }
            }
        }
        inline double qderiv(user_id j, user_id u, unsigned w)
        {
            return dQ[j][u][w];
        }
        inline void set_wvec(vector<double> w)
        {
            wvec = w;
        }
}; 
#endif

