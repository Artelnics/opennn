cmake_minimum_required(VERSION 2.8.12)

project(opennn)

file(GLOB SOURCES *.cpp)

add_library(opennn ${SOURCES} "vit_embedding_layer.cpp" "vit_embedding_layer.h" "vit_multihead_attention_layer.h" "vit_multihead_attention_layer.cpp"     "vit_feed_forward_network_layer_3d.h" "vit_feed_forward_network_layer_3d.cpp" "vision_transformer.h" "vision_transformer.cpp"  )
