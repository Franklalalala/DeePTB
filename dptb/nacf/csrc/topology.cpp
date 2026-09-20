// NACF periodic third-centre topology. Neighbor enumeration uses the vendored
// MIT-licensed Tonari core; see vendor/tonari/LICENSE and THIRD_PARTY.md.
#include "vendor/tonari/neighbors_cpu.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {
using I = int64_t;
using Key = std::array<I, 5>;
using Centre = std::array<I, 4>;
template<class K> struct Hash {
    size_t operator()(const K& k) const noexcept {
        size_t h = 0;
        for (auto x : k) h ^= std::hash<I>{}(x) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};
struct Plan {
    std::vector<Key> queries; // (AO, centre, AO image relative to centre)
    std::vector<std::array<I, 3>> terms; // (original edge row, left Q, right Q)
    std::vector<I> reverse;
    I broad_pairs = 0;
    double search_seconds = 0, join_seconds = 0;
};
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}
}

extern "C" {
int nacf_topology_abi() { return 1; }
void* nacf_topology_build_mode(I n, const double* pos, const double* cell,
                        const uint8_t* pbc, const double* ao_cut,
                        const double* centre_cut, I e, const I* edges,
                        const I* shifts, I max_terms, char* error, int mode) {
    try {
        if (n <= 0 || e < 0 || max_terms <= 0) throw std::invalid_argument("invalid topology size");
        if (mode < 0 || mode > 3) throw std::invalid_argument("invalid topology mode");
        auto out = std::make_unique<Plan>();
        std::unordered_map<Key, I, Hash<Key>> edge_rows;
        for (I r = 0; r < e; ++r) {
            Key key{edges[2*r], edges[2*r+1], shifts[3*r], shifts[3*r+1], shifts[3*r+2]};
            if (key[0] < 0 || key[0] >= n || key[1] < 0 || key[1] >= n)
                throw std::invalid_argument("edge atom index out of bounds");
            for (int a=0; a<3; ++a)
                if (!pbc[a] && key[a+2]) throw std::invalid_argument("shift in nonperiodic direction");
            if (key[0]==key[1] && key[2]==0 && key[3]==0 && key[4]==0)
                throw std::invalid_argument("onsite self-edge is not a hopping edge");
            if (!edge_rows.emplace(key, r).second) throw std::invalid_argument("duplicate edge");
        }
        out->reverse.resize(e);
        for (const auto& [key, row] : edge_rows) {
            auto it = edge_rows.find({key[1],key[0],-key[2],-key[3],-key[4]});
            if (it == edge_rows.end() && mode!=3) throw std::invalid_argument("every edge must have its reverse");
            out->reverse[row] = it == edge_rows.end() ? -1 : it->second;
        }
        const double radius = *std::max_element(ao_cut,ao_cut+n) +
                              *std::max_element(centre_cut,centre_cut+n) + 1e-9;
        auto t = Clock::now();
        const std::array<I, 2> batch{0,n};
        auto broad = neighbor_search::neighbor_list_cpu<double>(
            {pos,static_cast<size_t>(n*3)},batch,{cell,9},{pbc,3},radius,
            neighbor_search::PairMode::Full,neighbor_search::Algorithm::Auto,1);
        std::vector<I> pairs(broad.pair_count*2);
        std::vector<int32_t> images(broad.pair_count*3);
        neighbor_search::copy_pair_buffers(broad,pairs,images,1);
        out->broad_pairs = broad.pair_count;
        out->search_seconds = elapsed(t);
        t = Clock::now();
        for (size_t r=0; r<broad.pair_count; ++r) {
            I i=pairs[2*r], k=pairs[2*r+1];
            double d2=0;
            for (int a=0; a<3; ++a) {
                double d=pos[3*k+a]-pos[3*i+a];
                for (int b=0; b<3; ++b) d+=images[3*r+b]*cell[3*b+a];
                d2+=d*d;
            }
            // Projectors include both endpoints; VNA excludes the origin
            // centre. A negative centre cutoff disables a projector species.
            const double distance=std::sqrt(d2), support=ao_cut[i]+centre_cut[k];
            if (centre_cut[k] >= 0 && (mode==1 || mode==3 ? distance <= support+1e-12 : distance < support-1e-12)) {
                const I sign=mode==3?1:-1;
                out->queries.push_back({i,k,sign*I(images[3*r]),sign*I(images[3*r+1]),sign*I(images[3*r+2])});
            }
        }
        if (mode==1)
            for (I i=0; i<n; ++i)
                if (centre_cut[i]>=0) out->queries.push_back({i,i,0,0,0});
        std::sort(out->queries.begin(),out->queries.end());
        if (std::adjacent_find(out->queries.begin(),out->queries.end()) != out->queries.end())
            throw std::runtime_error("duplicate neighbour identity");
        if (mode==3) {
            // Density queries use the NEIGHBOUR image relative to the AO.
            // Origin self is excluded by the broad search, other self images remain.
            if (static_cast<I>(out->queries.size())>max_terms)
                throw std::length_error("density query budget exceeded");
            std::vector<std::vector<I>> grouped(n);
            for (I q=0;q<static_cast<I>(out->queries.size());++q)
                grouped[out->queries[q][0]].push_back(q);
            for (I r=0;r<e;++r) for (I q:grouped[edges[2*r]]) {
                const auto& x=out->queries[q];
                if (x[1]==edges[2*r+1] && x[2]==shifts[3*r] &&
                    x[3]==shifts[3*r+1] && x[4]==shifts[3*r+2]) continue;
                if (static_cast<I>(out->terms.size())==max_terms)
                    throw std::length_error("density edge term budget exceeded");
                out->terms.push_back({r,q,q});
            }
            out->join_seconds=elapsed(t);
            return out.release();
        }
        std::vector<std::unordered_map<Centre,I,Hash<Centre>>> neighbours(n);
        std::vector<std::vector<I>> by_atom(n);
        for (I q=0; q<static_cast<I>(out->queries.size()); ++q) {
            const auto& x=out->queries[q];
            neighbours[x[0]].emplace(Centre{x[1],-x[2],-x[3],-x[4]},q);
            by_atom[x[0]].push_back(q);
        }
        const I blocks=mode==0?e:(mode==1?n+e:n);
        for (I r=0; r<blocks; ++r) {
            if (mode==0 && r>out->reverse[r]) continue;
            const I edge=mode==0?r:r-n;
            const bool onsite=mode!=0 && r<n;
            I i=onsite?r:edges[2*edge],j=onsite?r:edges[2*edge+1];
            bool left=by_atom[i].size()<=by_atom[j].size();
            for (I q : by_atom[left?i:j]) {
                const auto& x=out->queries[q];
                Centre other{x[1],-x[2],-x[3],-x[4]};
                if (!onsite)
                    for (int a=0; a<3; ++a) other[a+1]+=(left?-1:1)*shifts[3*edge+a];
                auto it=neighbours[left?j:i].find(other);
                if (it==neighbours[left?j:i].end()) continue;
                if (static_cast<I>(out->terms.size())==max_terms)
                    throw std::length_error("third-centre term budget exceeded; reduce structure batch or raise max_terms explicitly");
                out->terms.push_back({r,left?q:it->second,left?it->second:q});
            }
        }
        std::sort(out->terms.begin(),out->terms.end());
        out->join_seconds=elapsed(t);
        return out.release();
    } catch (const std::exception& ex) {
        std::snprintf(error,1024,"%s",ex.what());
        return nullptr;
    } catch (...) {
        std::snprintf(error,1024,"unknown native topology error");
        return nullptr;
    }
}
void* nacf_topology_build(I n, const double* pos, const double* cell,
                        const uint8_t* pbc, const double* ao_cut,
                        const double* centre_cut, I e, const I* edges,
                        const I* shifts, I max_terms, char* error) {
    return nacf_topology_build_mode(n,pos,cell,pbc,ao_cut,centre_cut,e,edges,shifts,max_terms,error,0);
}
I nacf_topology_count(void* p,int which) {
    const auto& x=*static_cast<Plan*>(p);
    return which==0?x.queries.size():which==1?x.terms.size():which==2?x.reverse.size():x.broad_pairs;
}
const I* nacf_topology_data(void* p,int which) {
    const auto& x=*static_cast<Plan*>(p);
    return which==0?reinterpret_cast<const I*>(x.queries.data()):
           which==1?reinterpret_cast<const I*>(x.terms.data()):x.reverse.data();
}
double nacf_topology_seconds(void* p,int which) {
    const auto& x=*static_cast<Plan*>(p); return which==0?x.search_seconds:x.join_seconds;
}
void nacf_topology_free(void* p) { delete static_cast<Plan*>(p); }
}
