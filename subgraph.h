#ifndef SUBGRAPH_H
#define SUBGRAPH_H

#include <iostream>
#include <set>
#include <algorithm>
#include <map>
#include <unordered_map>

#include "params.h"

class Subgraph
{
    typedef set<user_id> vset;
    typedef map<user_id, unsigned> datemap;
    typedef vector<user_id> vvec;

    typedef unordered_map<user_id, unsigned> map_v2u;
    typedef map<unsigned, user_id> map_u2v;

    typedef unordered_map<user_id, double> scores;
    typedef unordered_map<user_id, scores> edge_scores;

    private:
        const unsigned Timepoint;
        const user_id start;

        map_v2u v2u;
        map_u2v u2v;
        edge_scores Q;

    public:
        Graph *subgraph;
        vector<user_id> zerodeg;
        vvec positive, negative;
        scores mutual;
        datemap added;

    private:
        Graph* build_lgraph(Graph *graph, unsigned edges)
        {
            Graph *sub = new Graph(edges, v2u.size(), true); // true does allocation 
            unsigned vc = 0, ec = 0;
            // iterate in increasing order. start has index 0
            for (map_u2v::iterator it = u2v.begin(); it != u2v.end(); it++)
            {
                vc = it->first;

                sub->adjacency_offset[vc] = ec;
                Q[vc] = scores();
                unsigned prev = ec;
                for (Graph::iterator e = graph->iterate_outgoing_edges(it->second); !e.end(); e++)
                {
                    unsigned nei = (*e).v2;
                    if ((*e).data < Timepoint && v2u.count(nei) > 0)
                    {
                        sub->adjacency_data[ec].v2 = v2u[nei];
                        sub->adjacency_data[ec].data = (*e).data;
                        ec++;
                        Q[vc][ v2u[nei] ] = 0.0;
                    }
                }
                if (prev == ec)
                    zerodeg.push_back(vc);
            }
            sub->adjacency_offset[vc+1] = ec;
            return sub;
        }

        Graph* create_subgraph(Graph *graph)
        {
            unsigned edges = 0, vertices = 0;
            vset friends, fofs, future;

            mutual[vertices] = 0;
            u2v[vertices] = start;
            v2u[start] = vertices++;
            for (Graph::iterator e = graph->iterate_outgoing_edges(start); !e.end(); e++) 
            {
                user_id fr = (*e).v2;
                if ((*e).data < Timepoint)
                {
                    u2v[vertices] = fr;
                    v2u[fr] = vertices++; 
                    friends.insert(fr);
                }
                else if ((*e).data > Timepoint)
                {
                    future.insert(fr);
                    added[fr] = (*e).data;
                }
            }
            edges += friends.size();
            // count initial friends of friends
            for (vset::iterator it = friends.begin(); it != friends.end(); it++)
            {
                user_id fr = *it;
                unsigned mut = 0;
                for (Graph::iterator e = graph->iterate_outgoing_edges(fr); !e.end(); e++) 
                {
                    if ((*e).data < Timepoint)
                    {
                        user_id fr = (*e).v2;
                        edges++;
                        if (fr != start)
                        {
                            if (friends.count(fr) == 0)
                                fofs.insert(fr);
                            else
                                mut++;
                        }
                    }
                }
                mutual[v2u[fr]] = mut;
            }
            // filter friends of friends
            vset filtered_fofs;
            for (vset::iterator it = fofs.begin(); it != fofs.end(); it++)
            {
                user_id fof = *it;
                unsigned mut = 0;
                for (Graph::iterator e = graph->iterate_outgoing_edges(fof); !e.end(); e++) 
                    if ((*e).data < Timepoint && friends.count((*e).v2) > 0)
                        mut++;
                if (mut >= 1)
                {
                    mutual[vertices] = mut;
                    u2v[vertices] = fof;
                    v2u[fof] = vertices++; 
                    filtered_fofs.insert(fof);
                }
            }
            // count filtered fof edges
            for (vset::iterator it = filtered_fofs.begin(); it != filtered_fofs.end(); it++)
            {
                user_id fof = *it;
                for (Graph::iterator e = graph->iterate_outgoing_edges(fof); !e.end(); e++) 
                    if ((*e).data < Timepoint && v2u.count((*e).v2) > 0)
                        edges++;
            }

            set_intersection(filtered_fofs.begin(), filtered_fofs.end(),
                             future.begin(),    future.end(),
                             inserter(positive, positive.end()));

            set_difference(filtered_fofs.begin(), filtered_fofs.end(),
                           positive.begin(),  positive.end(),
                           inserter(negative, negative.end()));
            //cout << "Vertex " << start << " neighbourhood. Fr = " << friends.size() << ", filter-fofs = " << filtered_fofs.size() 
            //   << ", edges = " << edges << ", pos = " << positive.size() << ", neg = " << negative.size() << "m1 = " << v2u.size() << ", m2 = " << u2v.size() << endl;
            //cout << friends.size() << "\t" << filtered_fofs.size() << "\t" << positive.size() << endl;
            return build_lgraph(graph, edges);
        }

    public:
        unsigned date_added(user_id id)
        {
            if (u2v.count(id) > 0 && added.count(u2v[id]) > 0)
                return added[ u2v[id] ];
            return 0;
        }

        unsigned total_added(unsigned from, unsigned until)
        {
            unsigned count = 0;
            for (datemap::iterator it = added.begin(); it != added.end(); it++)
                if (it->second >= from && it->second <= until)
                    count++;
            return count;
        }

        Subgraph(Graph *orig, user_id start, const unsigned T) : 
            Timepoint(T), start(start)
        {
            subgraph = create_subgraph(orig);
            for (unsigned i = 0; i < positive.size(); i++)
                positive[i] = v2u[ positive[i] ];
            for (unsigned i = 0; i < negative.size(); i++)
                negative[i] = v2u[ negative[i] ];
        }
        ~Subgraph()
        {
            delete subgraph;
        }
        inline double score(user_id from, user_id to)
        {
            return Q[from][to];
        }
        void recompute_scores(Params &p)
        {
            for (edge_scores::iterator it = Q.begin(); it != Q.end(); it++)
            {
                double sum = 0.0;
                for (scores::iterator e = it->second.begin(); e != it->second.end(); e++) 
                    sum += p.edge_strength(it->first, e->first);
                for (scores::iterator e = it->second.begin(); e != it->second.end(); e++) 
                    Q[it->first][e->first] = p.edge_strength(it->first, e->first) / sum;
            }
        }
};

#endif

