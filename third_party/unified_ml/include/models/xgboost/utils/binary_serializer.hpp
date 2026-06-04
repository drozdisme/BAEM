#pragma once
//                                        
//  Binary model serialisation  (P4-7)
//
//  Format (little-endian, fixed-width):
//  ┌                             ┐
//  │  MAGIC   4B  "XGB\x01"                                  │
//  │  VERSION 4B  uint32  (current = 1)                      │
//  │  HEADER  —   BoosterConfig fields (flat POD-like)       │
//  │  TREES   —   N × TreeBlock                              │
//  └                             ┘
//
//  TreeBlock:
//  ┌                             ┐
//  │  n_nodes  uint32                                        │
//  │  nodes[]  NodeRecord × n_nodes                          │
//  └                             ┘
//
//  NodeRecord (24 bytes each):
//  ┌                             ┐
//  │  node_id     int32                                      │
//  │  left_child  int32                                      │
//  │  right_child int32                                      │
//  │  feature_idx uint32                                     │
//  │  split_value float32                                    │
//  │  leaf_value  float32                                    │
//  └                             ┘
//                                        
#pragma once
#include "models/xgboost/core/types.hpp"
#include "models/xgboost/tree/tree_node.hpp"
#include <string>
#include <vector>
#include <cstdint>

namespace xgb {

class GradientBooster;

//   Constants                                 
constexpr uint32_t kBinaryMagic   = 0x01424758u; // "XGB\x01" LE
constexpr uint32_t kBinaryVersion = 1u;

//   Flat header written to file                         
#pragma pack(push, 1)
struct BinaryHeader {
    uint32_t magic        {kBinaryMagic};
    uint32_t version      {kBinaryVersion};
    uint32_t num_trees    {0};
    float    base_score   {0.5f};
    float    eta          {0.3f};
    uint32_t num_class    {1};
    char     objective[64]{};   // null-padded
    uint32_t reserved[8]  {};   // for future extension
};
static_assert(sizeof(BinaryHeader) == 4+4+4+4+4+4+64+32, "BinaryHeader size mismatch");

struct NodeRecord {
    int32_t  node_id     {-1};
    int32_t  left_child  {-1};
    int32_t  right_child {-1};
    uint32_t feature_idx {0};
    float    split_value {0.f};
    float    leaf_value  {0.f};
};
static_assert(sizeof(NodeRecord) == 24, "NodeRecord must be exactly 24 bytes");
#pragma pack(pop)

//   API                                     
void save_model_binary(const GradientBooster& booster, const std::string& path);
void load_model_binary(GradientBooster& booster, const std::string& path);

} // namespace xgb
