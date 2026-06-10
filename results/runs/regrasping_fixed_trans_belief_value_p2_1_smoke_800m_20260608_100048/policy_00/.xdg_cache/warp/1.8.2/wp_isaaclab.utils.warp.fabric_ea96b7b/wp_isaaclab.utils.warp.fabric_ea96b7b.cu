
#define WP_TILE_BLOCK_DIM 256
#define WP_NO_CRT
#include "builtin.h"

// avoid namespacing of float type for casting to float type, this is to avoid wp::float(x), which is not valid in C++
#define float(x) cast_float(x)
#define adj_float(x, adj_x, adj_ret) adj_cast_float(x, adj_x, adj_ret)

#define int(x) cast_int(x)
#define adj_int(x, adj_x, adj_ret) adj_cast_int(x, adj_x, adj_ret)

#define builtin_tid1d() wp::tid(_idx, dim)
#define builtin_tid2d(x, y) wp::tid(x, y, _idx, dim)
#define builtin_tid3d(x, y, z) wp::tid(x, y, z, _idx, dim)
#define builtin_tid4d(x, y, z, w) wp::tid(x, y, z, w, _idx, dim)

#define builtin_block_dim() wp::block_dim()

extern "C" {
}

// /root/autodl-tmp/wmdexpbt/runtime_root/src/IsaacLab/source/isaaclab/isaaclab/utils/warp/fabric.py:0
static CUDA_CALLABLE void _decompose_transformation_matrix_0(
    wp::mat_t<4, 4, wp::float32> var_m,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::quat_t<wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 3;
    const wp::int32 var_1 = 0;
    wp::float32 var_2;
    const wp::int32 var_3 = 3;
    const wp::int32 var_4 = 1;
    wp::float32 var_5;
    const wp::int32 var_6 = 3;
    const wp::int32 var_7 = 2;
    wp::float32 var_8;
    wp::vec_t<3, wp::float32> var_9;
    const wp::int32 var_10 = 0;
    const wp::int32 var_11 = 0;
    wp::float32 var_12;
    const wp::int32 var_13 = 0;
    const wp::int32 var_14 = 1;
    wp::float32 var_15;
    const wp::int32 var_16 = 0;
    const wp::int32 var_17 = 2;
    wp::float32 var_18;
    const wp::int32 var_19 = 1;
    const wp::int32 var_20 = 0;
    wp::float32 var_21;
    const wp::int32 var_22 = 1;
    const wp::int32 var_23 = 1;
    wp::float32 var_24;
    const wp::int32 var_25 = 1;
    const wp::int32 var_26 = 2;
    wp::float32 var_27;
    const wp::int32 var_28 = 2;
    const wp::int32 var_29 = 0;
    wp::float32 var_30;
    const wp::int32 var_31 = 2;
    const wp::int32 var_32 = 1;
    wp::float32 var_33;
    const wp::int32 var_34 = 2;
    const wp::int32 var_35 = 2;
    wp::float32 var_36;
    wp::float32 var_37;
    wp::float32 var_38;
    wp::float32 var_39;
    wp::float32 var_40;
    wp::float32 var_41;
    wp::float32 var_42;
    wp::float32 var_43;
    wp::float32 var_44;
    wp::float32 var_45;
    wp::float32 var_46;
    wp::float32 var_47;
    wp::float32 var_48;
    wp::float32 var_49;
    wp::float32 var_50;
    wp::float32 var_51;
    wp::float32 var_52;
    wp::float32 var_53;
    wp::float32 var_54;
    const wp::float32 var_55 = 0.0;
    bool var_56;
    wp::float32 var_57;
    wp::float32 var_58;
    wp::float32 var_59;
    wp::float32 var_60;
    wp::float32 var_61;
    wp::float32 var_62;
    const wp::float32 var_63 = 0.0;
    bool var_64;
    wp::float32 var_65;
    wp::float32 var_66;
    wp::float32 var_67;
    wp::float32 var_68;
    wp::float32 var_69;
    wp::float32 var_70;
    const wp::float32 var_71 = 0.0;
    bool var_72;
    wp::float32 var_73;
    wp::float32 var_74;
    wp::float32 var_75;
    wp::float32 var_76;
    wp::float32 var_77;
    wp::float32 var_78;
    wp::mat_t<3, 3, wp::float32> var_79;
    wp::mat_t<3, 3, wp::float32> var_80;
    wp::quat_t<wp::float32> var_81;
    wp::vec_t<3, wp::float32> var_82;
    //---------
    // forward
    // def _decompose_transformation_matrix(m: Any):  # -> tuple[wp.vec3f, wp.quatf, wp.vec3f]       <L 1>
    // position = wp.vec3f(m[3, 0], m[3, 1], m[3, 2])                                         <L 11>
    var_2 = wp::extract(var_m, var_0, var_1);
    var_5 = wp::extract(var_m, var_3, var_4);
    var_8 = wp::extract(var_m, var_6, var_7);
    var_9 = wp::vec_t<3, wp::float32>(var_2, var_5, var_8);
    // r00, r01, r02 = m[0, 0], m[0, 1], m[0, 2]                                              <L 13>
    var_12 = wp::extract(var_m, var_10, var_11);
    var_15 = wp::extract(var_m, var_13, var_14);
    var_18 = wp::extract(var_m, var_16, var_17);
    // r10, r11, r12 = m[1, 0], m[1, 1], m[1, 2]                                              <L 14>
    var_21 = wp::extract(var_m, var_19, var_20);
    var_24 = wp::extract(var_m, var_22, var_23);
    var_27 = wp::extract(var_m, var_25, var_26);
    // r20, r21, r22 = m[2, 0], m[2, 1], m[2, 2]                                              <L 15>
    var_30 = wp::extract(var_m, var_28, var_29);
    var_33 = wp::extract(var_m, var_31, var_32);
    var_36 = wp::extract(var_m, var_34, var_35);
    // sx = wp.sqrt(r00 * r00 + r01 * r01 + r02 * r02)                                        <L 17>
    var_37 = wp::mul(var_12, var_12);
    var_38 = wp::mul(var_15, var_15);
    var_39 = wp::add(var_37, var_38);
    var_40 = wp::mul(var_18, var_18);
    var_41 = wp::add(var_39, var_40);
    var_42 = wp::sqrt(var_41);
    // sy = wp.sqrt(r10 * r10 + r11 * r11 + r12 * r12)                                        <L 18>
    var_43 = wp::mul(var_21, var_21);
    var_44 = wp::mul(var_24, var_24);
    var_45 = wp::add(var_43, var_44);
    var_46 = wp::mul(var_27, var_27);
    var_47 = wp::add(var_45, var_46);
    var_48 = wp::sqrt(var_47);
    // sz = wp.sqrt(r20 * r20 + r21 * r21 + r22 * r22)                                        <L 19>
    var_49 = wp::mul(var_30, var_30);
    var_50 = wp::mul(var_33, var_33);
    var_51 = wp::add(var_49, var_50);
    var_52 = wp::mul(var_36, var_36);
    var_53 = wp::add(var_51, var_52);
    var_54 = wp::sqrt(var_53);
    // if sx != 0.0:                                                                          <L 21>
    var_56 = (var_42 != var_55);
    if (var_56) {
        // r00 /= sx                                                                          <L 22>
        var_57 = wp::div(var_12, var_42);
        // r01 /= sx                                                                          <L 23>
        var_58 = wp::div(var_15, var_42);
        // r02 /= sx                                                                          <L 24>
        var_59 = wp::div(var_18, var_42);
    }
    var_60 = wp::where(var_56, var_57, var_12);
    var_61 = wp::where(var_56, var_58, var_15);
    var_62 = wp::where(var_56, var_59, var_18);
    // if sy != 0.0:                                                                          <L 25>
    var_64 = (var_48 != var_63);
    if (var_64) {
        // r10 /= sy                                                                          <L 26>
        var_65 = wp::div(var_21, var_48);
        // r11 /= sy                                                                          <L 27>
        var_66 = wp::div(var_24, var_48);
        // r12 /= sy                                                                          <L 28>
        var_67 = wp::div(var_27, var_48);
    }
    var_68 = wp::where(var_64, var_65, var_21);
    var_69 = wp::where(var_64, var_66, var_24);
    var_70 = wp::where(var_64, var_67, var_27);
    // if sz != 0.0:                                                                          <L 29>
    var_72 = (var_54 != var_71);
    if (var_72) {
        // r20 /= sz                                                                          <L 30>
        var_73 = wp::div(var_30, var_54);
        // r21 /= sz                                                                          <L 31>
        var_74 = wp::div(var_33, var_54);
        // r22 /= sz                                                                          <L 32>
        var_75 = wp::div(var_36, var_54);
    }
    var_76 = wp::where(var_72, var_73, var_30);
    var_77 = wp::where(var_72, var_74, var_33);
    var_78 = wp::where(var_72, var_75, var_36);
    // rotation = wp.quat_from_matrix(  # type: ignore[arg-type]                              <L 34>
    // wp.transpose(wp.mat33f(r00, r01, r02, r10, r11, r12, r20, r21, r22))  # type: ignore[arg-type]       <L 35>
    var_79 = wp::mat_t<3, 3, wp::float32>(var_60, var_61, var_62, var_68, var_69, var_70, var_76, var_77, var_78);
    var_80 = wp::transpose(var_79);
    var_81 = wp::quat_from_matrix(var_80);
    // scale = wp.vec3f(sx, sy, sz)                                                           <L 38>
    var_82 = wp::vec_t<3, wp::float32>(var_42, var_48, var_54);
    // return position, rotation, scale                                                       <L 39>
    ret_0 = var_9;
    ret_1 = var_81;
    ret_2 = var_82;
    return;
}


// /root/autodl-tmp/wmdexpbt/runtime_root/src/IsaacLab/source/isaaclab/isaaclab/utils/warp/fabric.py:0
static CUDA_CALLABLE void adj__decompose_transformation_matrix_0(
    wp::mat_t<4, 4, wp::float32> var_m,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::quat_t<wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::mat_t<4, 4, wp::float32> & adj_m,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::quat_t<wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 3;
    const wp::int32 var_1 = 0;
    wp::float32 var_2;
    const wp::int32 var_3 = 3;
    const wp::int32 var_4 = 1;
    wp::float32 var_5;
    const wp::int32 var_6 = 3;
    const wp::int32 var_7 = 2;
    wp::float32 var_8;
    wp::vec_t<3, wp::float32> var_9;
    const wp::int32 var_10 = 0;
    const wp::int32 var_11 = 0;
    wp::float32 var_12;
    const wp::int32 var_13 = 0;
    const wp::int32 var_14 = 1;
    wp::float32 var_15;
    const wp::int32 var_16 = 0;
    const wp::int32 var_17 = 2;
    wp::float32 var_18;
    const wp::int32 var_19 = 1;
    const wp::int32 var_20 = 0;
    wp::float32 var_21;
    const wp::int32 var_22 = 1;
    const wp::int32 var_23 = 1;
    wp::float32 var_24;
    const wp::int32 var_25 = 1;
    const wp::int32 var_26 = 2;
    wp::float32 var_27;
    const wp::int32 var_28 = 2;
    const wp::int32 var_29 = 0;
    wp::float32 var_30;
    const wp::int32 var_31 = 2;
    const wp::int32 var_32 = 1;
    wp::float32 var_33;
    const wp::int32 var_34 = 2;
    const wp::int32 var_35 = 2;
    wp::float32 var_36;
    wp::float32 var_37;
    wp::float32 var_38;
    wp::float32 var_39;
    wp::float32 var_40;
    wp::float32 var_41;
    wp::float32 var_42;
    wp::float32 var_43;
    wp::float32 var_44;
    wp::float32 var_45;
    wp::float32 var_46;
    wp::float32 var_47;
    wp::float32 var_48;
    wp::float32 var_49;
    wp::float32 var_50;
    wp::float32 var_51;
    wp::float32 var_52;
    wp::float32 var_53;
    wp::float32 var_54;
    const wp::float32 var_55 = 0.0;
    bool var_56;
    wp::float32 var_57;
    wp::float32 var_58;
    wp::float32 var_59;
    wp::float32 var_60;
    wp::float32 var_61;
    wp::float32 var_62;
    const wp::float32 var_63 = 0.0;
    bool var_64;
    wp::float32 var_65;
    wp::float32 var_66;
    wp::float32 var_67;
    wp::float32 var_68;
    wp::float32 var_69;
    wp::float32 var_70;
    const wp::float32 var_71 = 0.0;
    bool var_72;
    wp::float32 var_73;
    wp::float32 var_74;
    wp::float32 var_75;
    wp::float32 var_76;
    wp::float32 var_77;
    wp::float32 var_78;
    wp::mat_t<3, 3, wp::float32> var_79;
    wp::mat_t<3, 3, wp::float32> var_80;
    wp::quat_t<wp::float32> var_81;
    wp::vec_t<3, wp::float32> var_82;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::int32 adj_1 = {};
    wp::float32 adj_2 = {};
    wp::int32 adj_3 = {};
    wp::int32 adj_4 = {};
    wp::float32 adj_5 = {};
    wp::int32 adj_6 = {};
    wp::int32 adj_7 = {};
    wp::float32 adj_8 = {};
    wp::vec_t<3, wp::float32> adj_9 = {};
    wp::int32 adj_10 = {};
    wp::int32 adj_11 = {};
    wp::float32 adj_12 = {};
    wp::int32 adj_13 = {};
    wp::int32 adj_14 = {};
    wp::float32 adj_15 = {};
    wp::int32 adj_16 = {};
    wp::int32 adj_17 = {};
    wp::float32 adj_18 = {};
    wp::int32 adj_19 = {};
    wp::int32 adj_20 = {};
    wp::float32 adj_21 = {};
    wp::int32 adj_22 = {};
    wp::int32 adj_23 = {};
    wp::float32 adj_24 = {};
    wp::int32 adj_25 = {};
    wp::int32 adj_26 = {};
    wp::float32 adj_27 = {};
    wp::int32 adj_28 = {};
    wp::int32 adj_29 = {};
    wp::float32 adj_30 = {};
    wp::int32 adj_31 = {};
    wp::int32 adj_32 = {};
    wp::float32 adj_33 = {};
    wp::int32 adj_34 = {};
    wp::int32 adj_35 = {};
    wp::float32 adj_36 = {};
    wp::float32 adj_37 = {};
    wp::float32 adj_38 = {};
    wp::float32 adj_39 = {};
    wp::float32 adj_40 = {};
    wp::float32 adj_41 = {};
    wp::float32 adj_42 = {};
    wp::float32 adj_43 = {};
    wp::float32 adj_44 = {};
    wp::float32 adj_45 = {};
    wp::float32 adj_46 = {};
    wp::float32 adj_47 = {};
    wp::float32 adj_48 = {};
    wp::float32 adj_49 = {};
    wp::float32 adj_50 = {};
    wp::float32 adj_51 = {};
    wp::float32 adj_52 = {};
    wp::float32 adj_53 = {};
    wp::float32 adj_54 = {};
    wp::float32 adj_55 = {};
    bool adj_56 = {};
    wp::float32 adj_57 = {};
    wp::float32 adj_58 = {};
    wp::float32 adj_59 = {};
    wp::float32 adj_60 = {};
    wp::float32 adj_61 = {};
    wp::float32 adj_62 = {};
    wp::float32 adj_63 = {};
    bool adj_64 = {};
    wp::float32 adj_65 = {};
    wp::float32 adj_66 = {};
    wp::float32 adj_67 = {};
    wp::float32 adj_68 = {};
    wp::float32 adj_69 = {};
    wp::float32 adj_70 = {};
    wp::float32 adj_71 = {};
    bool adj_72 = {};
    wp::float32 adj_73 = {};
    wp::float32 adj_74 = {};
    wp::float32 adj_75 = {};
    wp::float32 adj_76 = {};
    wp::float32 adj_77 = {};
    wp::float32 adj_78 = {};
    wp::mat_t<3, 3, wp::float32> adj_79 = {};
    wp::mat_t<3, 3, wp::float32> adj_80 = {};
    wp::quat_t<wp::float32> adj_81 = {};
    wp::vec_t<3, wp::float32> adj_82 = {};
    //---------
    // forward
    // def _decompose_transformation_matrix(m: Any):  # -> tuple[wp.vec3f, wp.quatf, wp.vec3f]       <L 1>
    // position = wp.vec3f(m[3, 0], m[3, 1], m[3, 2])                                         <L 11>
    var_2 = wp::extract(var_m, var_0, var_1);
    var_5 = wp::extract(var_m, var_3, var_4);
    var_8 = wp::extract(var_m, var_6, var_7);
    var_9 = wp::vec_t<3, wp::float32>(var_2, var_5, var_8);
    // r00, r01, r02 = m[0, 0], m[0, 1], m[0, 2]                                              <L 13>
    var_12 = wp::extract(var_m, var_10, var_11);
    var_15 = wp::extract(var_m, var_13, var_14);
    var_18 = wp::extract(var_m, var_16, var_17);
    // r10, r11, r12 = m[1, 0], m[1, 1], m[1, 2]                                              <L 14>
    var_21 = wp::extract(var_m, var_19, var_20);
    var_24 = wp::extract(var_m, var_22, var_23);
    var_27 = wp::extract(var_m, var_25, var_26);
    // r20, r21, r22 = m[2, 0], m[2, 1], m[2, 2]                                              <L 15>
    var_30 = wp::extract(var_m, var_28, var_29);
    var_33 = wp::extract(var_m, var_31, var_32);
    var_36 = wp::extract(var_m, var_34, var_35);
    // sx = wp.sqrt(r00 * r00 + r01 * r01 + r02 * r02)                                        <L 17>
    var_37 = wp::mul(var_12, var_12);
    var_38 = wp::mul(var_15, var_15);
    var_39 = wp::add(var_37, var_38);
    var_40 = wp::mul(var_18, var_18);
    var_41 = wp::add(var_39, var_40);
    var_42 = wp::sqrt(var_41);
    // sy = wp.sqrt(r10 * r10 + r11 * r11 + r12 * r12)                                        <L 18>
    var_43 = wp::mul(var_21, var_21);
    var_44 = wp::mul(var_24, var_24);
    var_45 = wp::add(var_43, var_44);
    var_46 = wp::mul(var_27, var_27);
    var_47 = wp::add(var_45, var_46);
    var_48 = wp::sqrt(var_47);
    // sz = wp.sqrt(r20 * r20 + r21 * r21 + r22 * r22)                                        <L 19>
    var_49 = wp::mul(var_30, var_30);
    var_50 = wp::mul(var_33, var_33);
    var_51 = wp::add(var_49, var_50);
    var_52 = wp::mul(var_36, var_36);
    var_53 = wp::add(var_51, var_52);
    var_54 = wp::sqrt(var_53);
    // if sx != 0.0:                                                                          <L 21>
    var_56 = (var_42 != var_55);
    if (var_56) {
        // r00 /= sx                                                                          <L 22>
        var_57 = wp::div(var_12, var_42);
        // r01 /= sx                                                                          <L 23>
        var_58 = wp::div(var_15, var_42);
        // r02 /= sx                                                                          <L 24>
        var_59 = wp::div(var_18, var_42);
    }
    var_60 = wp::where(var_56, var_57, var_12);
    var_61 = wp::where(var_56, var_58, var_15);
    var_62 = wp::where(var_56, var_59, var_18);
    // if sy != 0.0:                                                                          <L 25>
    var_64 = (var_48 != var_63);
    if (var_64) {
        // r10 /= sy                                                                          <L 26>
        var_65 = wp::div(var_21, var_48);
        // r11 /= sy                                                                          <L 27>
        var_66 = wp::div(var_24, var_48);
        // r12 /= sy                                                                          <L 28>
        var_67 = wp::div(var_27, var_48);
    }
    var_68 = wp::where(var_64, var_65, var_21);
    var_69 = wp::where(var_64, var_66, var_24);
    var_70 = wp::where(var_64, var_67, var_27);
    // if sz != 0.0:                                                                          <L 29>
    var_72 = (var_54 != var_71);
    if (var_72) {
        // r20 /= sz                                                                          <L 30>
        var_73 = wp::div(var_30, var_54);
        // r21 /= sz                                                                          <L 31>
        var_74 = wp::div(var_33, var_54);
        // r22 /= sz                                                                          <L 32>
        var_75 = wp::div(var_36, var_54);
    }
    var_76 = wp::where(var_72, var_73, var_30);
    var_77 = wp::where(var_72, var_74, var_33);
    var_78 = wp::where(var_72, var_75, var_36);
    // rotation = wp.quat_from_matrix(  # type: ignore[arg-type]                              <L 34>
    // wp.transpose(wp.mat33f(r00, r01, r02, r10, r11, r12, r20, r21, r22))  # type: ignore[arg-type]       <L 35>
    var_79 = wp::mat_t<3, 3, wp::float32>(var_60, var_61, var_62, var_68, var_69, var_70, var_76, var_77, var_78);
    var_80 = wp::transpose(var_79);
    var_81 = wp::quat_from_matrix(var_80);
    // scale = wp.vec3f(sx, sy, sz)                                                           <L 38>
    var_82 = wp::vec_t<3, wp::float32>(var_42, var_48, var_54);
    // return position, rotation, scale                                                       <L 39>
    ret_0 = var_9;
    ret_1 = var_81;
    ret_2 = var_82;
    goto label0;
    //---------
    // reverse
    label0:;
    adj_82 += adj_ret_2;
    adj_81 += adj_ret_1;
    adj_9 += adj_ret_0;
    // adj: return position, rotation, scale                                                  <L 39>
    wp::adj_vec_t(var_42, var_48, var_54, adj_42, adj_48, adj_54, adj_82);
    // adj: scale = wp.vec3f(sx, sy, sz)                                                      <L 38>
    wp::adj_quat_from_matrix(var_80, adj_80, adj_81);
    wp::adj_transpose(var_79, adj_79, adj_80);
    wp::adj_mat_t(var_60, var_61, var_62, var_68, var_69, var_70, var_76, var_77, var_78, adj_60, adj_61, adj_62, adj_68, adj_69, adj_70, adj_76, adj_77, adj_78, adj_79);
    // adj: wp.transpose(wp.mat33f(r00, r01, r02, r10, r11, r12, r20, r21, r22))  # type: ignore[arg-type]  <L 35>
    // adj: rotation = wp.quat_from_matrix(  # type: ignore[arg-type]                         <L 34>
    wp::adj_where(var_72, var_75, var_36, adj_72, adj_75, adj_36, adj_78);
    wp::adj_where(var_72, var_74, var_33, adj_72, adj_74, adj_33, adj_77);
    wp::adj_where(var_72, var_73, var_30, adj_72, adj_73, adj_30, adj_76);
    if (var_72) {
        wp::adj_div(var_36, var_54, var_75, adj_36, adj_54, adj_75);
        // adj: r22 /= sz                                                                     <L 32>
        wp::adj_div(var_33, var_54, var_74, adj_33, adj_54, adj_74);
        // adj: r21 /= sz                                                                     <L 31>
        wp::adj_div(var_30, var_54, var_73, adj_30, adj_54, adj_73);
        // adj: r20 /= sz                                                                     <L 30>
    }
    // adj: if sz != 0.0:                                                                     <L 29>
    wp::adj_where(var_64, var_67, var_27, adj_64, adj_67, adj_27, adj_70);
    wp::adj_where(var_64, var_66, var_24, adj_64, adj_66, adj_24, adj_69);
    wp::adj_where(var_64, var_65, var_21, adj_64, adj_65, adj_21, adj_68);
    if (var_64) {
        wp::adj_div(var_27, var_48, var_67, adj_27, adj_48, adj_67);
        // adj: r12 /= sy                                                                     <L 28>
        wp::adj_div(var_24, var_48, var_66, adj_24, adj_48, adj_66);
        // adj: r11 /= sy                                                                     <L 27>
        wp::adj_div(var_21, var_48, var_65, adj_21, adj_48, adj_65);
        // adj: r10 /= sy                                                                     <L 26>
    }
    // adj: if sy != 0.0:                                                                     <L 25>
    wp::adj_where(var_56, var_59, var_18, adj_56, adj_59, adj_18, adj_62);
    wp::adj_where(var_56, var_58, var_15, adj_56, adj_58, adj_15, adj_61);
    wp::adj_where(var_56, var_57, var_12, adj_56, adj_57, adj_12, adj_60);
    if (var_56) {
        wp::adj_div(var_18, var_42, var_59, adj_18, adj_42, adj_59);
        // adj: r02 /= sx                                                                     <L 24>
        wp::adj_div(var_15, var_42, var_58, adj_15, adj_42, adj_58);
        // adj: r01 /= sx                                                                     <L 23>
        wp::adj_div(var_12, var_42, var_57, adj_12, adj_42, adj_57);
        // adj: r00 /= sx                                                                     <L 22>
    }
    // adj: if sx != 0.0:                                                                     <L 21>
    wp::adj_sqrt(var_53, var_54, adj_53, adj_54);
    wp::adj_add(var_51, var_52, adj_51, adj_52, adj_53);
    wp::adj_mul(var_36, var_36, adj_36, adj_36, adj_52);
    wp::adj_add(var_49, var_50, adj_49, adj_50, adj_51);
    wp::adj_mul(var_33, var_33, adj_33, adj_33, adj_50);
    wp::adj_mul(var_30, var_30, adj_30, adj_30, adj_49);
    // adj: sz = wp.sqrt(r20 * r20 + r21 * r21 + r22 * r22)                                   <L 19>
    wp::adj_sqrt(var_47, var_48, adj_47, adj_48);
    wp::adj_add(var_45, var_46, adj_45, adj_46, adj_47);
    wp::adj_mul(var_27, var_27, adj_27, adj_27, adj_46);
    wp::adj_add(var_43, var_44, adj_43, adj_44, adj_45);
    wp::adj_mul(var_24, var_24, adj_24, adj_24, adj_44);
    wp::adj_mul(var_21, var_21, adj_21, adj_21, adj_43);
    // adj: sy = wp.sqrt(r10 * r10 + r11 * r11 + r12 * r12)                                   <L 18>
    wp::adj_sqrt(var_41, var_42, adj_41, adj_42);
    wp::adj_add(var_39, var_40, adj_39, adj_40, adj_41);
    wp::adj_mul(var_18, var_18, adj_18, adj_18, adj_40);
    wp::adj_add(var_37, var_38, adj_37, adj_38, adj_39);
    wp::adj_mul(var_15, var_15, adj_15, adj_15, adj_38);
    wp::adj_mul(var_12, var_12, adj_12, adj_12, adj_37);
    // adj: sx = wp.sqrt(r00 * r00 + r01 * r01 + r02 * r02)                                   <L 17>
    wp::adj_extract(var_m, var_34, var_35, adj_m, adj_34, adj_35, adj_36);
    wp::adj_extract(var_m, var_31, var_32, adj_m, adj_31, adj_32, adj_33);
    wp::adj_extract(var_m, var_28, var_29, adj_m, adj_28, adj_29, adj_30);
    // adj: r20, r21, r22 = m[2, 0], m[2, 1], m[2, 2]                                         <L 15>
    wp::adj_extract(var_m, var_25, var_26, adj_m, adj_25, adj_26, adj_27);
    wp::adj_extract(var_m, var_22, var_23, adj_m, adj_22, adj_23, adj_24);
    wp::adj_extract(var_m, var_19, var_20, adj_m, adj_19, adj_20, adj_21);
    // adj: r10, r11, r12 = m[1, 0], m[1, 1], m[1, 2]                                         <L 14>
    wp::adj_extract(var_m, var_16, var_17, adj_m, adj_16, adj_17, adj_18);
    wp::adj_extract(var_m, var_13, var_14, adj_m, adj_13, adj_14, adj_15);
    wp::adj_extract(var_m, var_10, var_11, adj_m, adj_10, adj_11, adj_12);
    // adj: r00, r01, r02 = m[0, 0], m[0, 1], m[0, 2]                                         <L 13>
    wp::adj_vec_t(var_2, var_5, var_8, adj_2, adj_5, adj_8, adj_9);
    wp::adj_extract(var_m, var_6, var_7, adj_m, adj_6, adj_7, adj_8);
    wp::adj_extract(var_m, var_3, var_4, adj_m, adj_3, adj_4, adj_5);
    wp::adj_extract(var_m, var_0, var_1, adj_m, adj_0, adj_1, adj_2);
    // adj: position = wp.vec3f(m[3, 0], m[3, 1], m[3, 2])                                    <L 11>
    // adj: def _decompose_transformation_matrix(m: Any):  # -> tuple[wp.vec3f, wp.quatf, wp.vec3f]  <L 1>
    return;
}


// /root/autodl-tmp/wmdexpbt/runtime_root/envs/isaaclab311/lib/python3.11/site-packages/isaacsim/kit/data/Kit/Isaac-Sim/5.1/exts/3/omni.warp.core-1.8.2+lx64/warp/math.py:248
static CUDA_CALLABLE wp::mat_t<4, 4, wp::float32> transform_compose_0(
    wp::vec_t<3, wp::float32> var_position,
    wp::quat_t<wp::float32> var_rotation,
    wp::vec_t<3, wp::float32> var_scale)
{
    //---------
    // primal vars
    wp::mat_t<3, 3, wp::float32> var_0;
    const wp::int32 var_1 = 0;
    wp::float32 var_2;
    const wp::int32 var_3 = 0;
    const wp::int32 var_4 = 0;
    wp::float32 var_5;
    wp::float32 var_6;
    const wp::int32 var_7 = 1;
    wp::float32 var_8;
    const wp::int32 var_9 = 0;
    const wp::int32 var_10 = 1;
    wp::float32 var_11;
    wp::float32 var_12;
    const wp::int32 var_13 = 2;
    wp::float32 var_14;
    const wp::int32 var_15 = 0;
    const wp::int32 var_16 = 2;
    wp::float32 var_17;
    wp::float32 var_18;
    const wp::int32 var_19 = 0;
    wp::float32 var_20;
    const wp::int32 var_21 = 0;
    wp::float32 var_22;
    const wp::int32 var_23 = 1;
    const wp::int32 var_24 = 0;
    wp::float32 var_25;
    wp::float32 var_26;
    const wp::int32 var_27 = 1;
    wp::float32 var_28;
    const wp::int32 var_29 = 1;
    const wp::int32 var_30 = 1;
    wp::float32 var_31;
    wp::float32 var_32;
    const wp::int32 var_33 = 2;
    wp::float32 var_34;
    const wp::int32 var_35 = 1;
    const wp::int32 var_36 = 2;
    wp::float32 var_37;
    wp::float32 var_38;
    const wp::int32 var_39 = 1;
    wp::float32 var_40;
    const wp::int32 var_41 = 0;
    wp::float32 var_42;
    const wp::int32 var_43 = 2;
    const wp::int32 var_44 = 0;
    wp::float32 var_45;
    wp::float32 var_46;
    const wp::int32 var_47 = 1;
    wp::float32 var_48;
    const wp::int32 var_49 = 2;
    const wp::int32 var_50 = 1;
    wp::float32 var_51;
    wp::float32 var_52;
    const wp::int32 var_53 = 2;
    wp::float32 var_54;
    const wp::int32 var_55 = 2;
    const wp::int32 var_56 = 2;
    wp::float32 var_57;
    wp::float32 var_58;
    const wp::int32 var_59 = 2;
    wp::float32 var_60;
    const wp::float32 var_61 = 0.0;
    wp::float32 var_62;
    const wp::float32 var_63 = 0.0;
    wp::float32 var_64;
    const wp::float32 var_65 = 0.0;
    wp::float32 var_66;
    const wp::float32 var_67 = 1.0;
    wp::float32 var_68;
    wp::mat_t<4, 4, wp::float32> var_69;
    //---------
    // forward
    // def transform_compose(position: vec3, rotation: quat, scale: vec3):                    <L 248>
    // R = wp.quat_to_matrix(rotation)                                                        <L 274>
    var_0 = wp::quat_to_matrix(var_rotation);
    // return mat44(                                                                          <L 276>
    // scale[0] * R[0,0], scale[1] * R[0,1], scale[2] * R[0,2], position[0],                  <L 277>
    var_2 = wp::extract(var_scale, var_1);
    var_5 = wp::extract(var_0, var_3, var_4);
    var_6 = wp::mul(var_2, var_5);
    var_8 = wp::extract(var_scale, var_7);
    var_11 = wp::extract(var_0, var_9, var_10);
    var_12 = wp::mul(var_8, var_11);
    var_14 = wp::extract(var_scale, var_13);
    var_17 = wp::extract(var_0, var_15, var_16);
    var_18 = wp::mul(var_14, var_17);
    var_20 = wp::extract(var_position, var_19);
    // scale[0] * R[1,0], scale[1] * R[1,1], scale[2] * R[1,2], position[1],                  <L 278>
    var_22 = wp::extract(var_scale, var_21);
    var_25 = wp::extract(var_0, var_23, var_24);
    var_26 = wp::mul(var_22, var_25);
    var_28 = wp::extract(var_scale, var_27);
    var_31 = wp::extract(var_0, var_29, var_30);
    var_32 = wp::mul(var_28, var_31);
    var_34 = wp::extract(var_scale, var_33);
    var_37 = wp::extract(var_0, var_35, var_36);
    var_38 = wp::mul(var_34, var_37);
    var_40 = wp::extract(var_position, var_39);
    // scale[0] * R[2,0], scale[1] * R[2,1], scale[2] * R[2,2], position[2],                  <L 279>
    var_42 = wp::extract(var_scale, var_41);
    var_45 = wp::extract(var_0, var_43, var_44);
    var_46 = wp::mul(var_42, var_45);
    var_48 = wp::extract(var_scale, var_47);
    var_51 = wp::extract(var_0, var_49, var_50);
    var_52 = wp::mul(var_48, var_51);
    var_54 = wp::extract(var_scale, var_53);
    var_57 = wp::extract(var_0, var_55, var_56);
    var_58 = wp::mul(var_54, var_57);
    var_60 = wp::extract(var_position, var_59);
    // dtype(0.0), dtype(0.0), dtype(0.0), dtype(1.0),                                        <L 280>
    var_62 = wp::float32(var_61);
    var_64 = wp::float32(var_63);
    var_66 = wp::float32(var_65);
    var_68 = wp::float32(var_67);
    var_69 = wp::mat_t<4, 4, wp::float32>(var_6, var_12, var_18, var_20, var_26, var_32, var_38, var_40, var_46, var_52, var_58, var_60, var_62, var_64, var_66, var_68);
    return var_69;
}


// /root/autodl-tmp/wmdexpbt/runtime_root/envs/isaaclab311/lib/python3.11/site-packages/isaacsim/kit/data/Kit/Isaac-Sim/5.1/exts/3/omni.warp.core-1.8.2+lx64/warp/math.py:248
static CUDA_CALLABLE void adj_transform_compose_0(
    wp::vec_t<3, wp::float32> var_position,
    wp::quat_t<wp::float32> var_rotation,
    wp::vec_t<3, wp::float32> var_scale,
    wp::vec_t<3, wp::float32> & adj_position,
    wp::quat_t<wp::float32> & adj_rotation,
    wp::vec_t<3, wp::float32> & adj_scale,
    wp::mat_t<4, 4, wp::float32> & adj_ret)
{
    //---------
    // primal vars
    wp::mat_t<3, 3, wp::float32> var_0;
    const wp::int32 var_1 = 0;
    wp::float32 var_2;
    const wp::int32 var_3 = 0;
    const wp::int32 var_4 = 0;
    wp::float32 var_5;
    wp::float32 var_6;
    const wp::int32 var_7 = 1;
    wp::float32 var_8;
    const wp::int32 var_9 = 0;
    const wp::int32 var_10 = 1;
    wp::float32 var_11;
    wp::float32 var_12;
    const wp::int32 var_13 = 2;
    wp::float32 var_14;
    const wp::int32 var_15 = 0;
    const wp::int32 var_16 = 2;
    wp::float32 var_17;
    wp::float32 var_18;
    const wp::int32 var_19 = 0;
    wp::float32 var_20;
    const wp::int32 var_21 = 0;
    wp::float32 var_22;
    const wp::int32 var_23 = 1;
    const wp::int32 var_24 = 0;
    wp::float32 var_25;
    wp::float32 var_26;
    const wp::int32 var_27 = 1;
    wp::float32 var_28;
    const wp::int32 var_29 = 1;
    const wp::int32 var_30 = 1;
    wp::float32 var_31;
    wp::float32 var_32;
    const wp::int32 var_33 = 2;
    wp::float32 var_34;
    const wp::int32 var_35 = 1;
    const wp::int32 var_36 = 2;
    wp::float32 var_37;
    wp::float32 var_38;
    const wp::int32 var_39 = 1;
    wp::float32 var_40;
    const wp::int32 var_41 = 0;
    wp::float32 var_42;
    const wp::int32 var_43 = 2;
    const wp::int32 var_44 = 0;
    wp::float32 var_45;
    wp::float32 var_46;
    const wp::int32 var_47 = 1;
    wp::float32 var_48;
    const wp::int32 var_49 = 2;
    const wp::int32 var_50 = 1;
    wp::float32 var_51;
    wp::float32 var_52;
    const wp::int32 var_53 = 2;
    wp::float32 var_54;
    const wp::int32 var_55 = 2;
    const wp::int32 var_56 = 2;
    wp::float32 var_57;
    wp::float32 var_58;
    const wp::int32 var_59 = 2;
    wp::float32 var_60;
    const wp::float32 var_61 = 0.0;
    wp::float32 var_62;
    const wp::float32 var_63 = 0.0;
    wp::float32 var_64;
    const wp::float32 var_65 = 0.0;
    wp::float32 var_66;
    const wp::float32 var_67 = 1.0;
    wp::float32 var_68;
    wp::mat_t<4, 4, wp::float32> var_69;
    //---------
    // dual vars
    wp::mat_t<3, 3, wp::float32> adj_0 = {};
    wp::int32 adj_1 = {};
    wp::float32 adj_2 = {};
    wp::int32 adj_3 = {};
    wp::int32 adj_4 = {};
    wp::float32 adj_5 = {};
    wp::float32 adj_6 = {};
    wp::int32 adj_7 = {};
    wp::float32 adj_8 = {};
    wp::int32 adj_9 = {};
    wp::int32 adj_10 = {};
    wp::float32 adj_11 = {};
    wp::float32 adj_12 = {};
    wp::int32 adj_13 = {};
    wp::float32 adj_14 = {};
    wp::int32 adj_15 = {};
    wp::int32 adj_16 = {};
    wp::float32 adj_17 = {};
    wp::float32 adj_18 = {};
    wp::int32 adj_19 = {};
    wp::float32 adj_20 = {};
    wp::int32 adj_21 = {};
    wp::float32 adj_22 = {};
    wp::int32 adj_23 = {};
    wp::int32 adj_24 = {};
    wp::float32 adj_25 = {};
    wp::float32 adj_26 = {};
    wp::int32 adj_27 = {};
    wp::float32 adj_28 = {};
    wp::int32 adj_29 = {};
    wp::int32 adj_30 = {};
    wp::float32 adj_31 = {};
    wp::float32 adj_32 = {};
    wp::int32 adj_33 = {};
    wp::float32 adj_34 = {};
    wp::int32 adj_35 = {};
    wp::int32 adj_36 = {};
    wp::float32 adj_37 = {};
    wp::float32 adj_38 = {};
    wp::int32 adj_39 = {};
    wp::float32 adj_40 = {};
    wp::int32 adj_41 = {};
    wp::float32 adj_42 = {};
    wp::int32 adj_43 = {};
    wp::int32 adj_44 = {};
    wp::float32 adj_45 = {};
    wp::float32 adj_46 = {};
    wp::int32 adj_47 = {};
    wp::float32 adj_48 = {};
    wp::int32 adj_49 = {};
    wp::int32 adj_50 = {};
    wp::float32 adj_51 = {};
    wp::float32 adj_52 = {};
    wp::int32 adj_53 = {};
    wp::float32 adj_54 = {};
    wp::int32 adj_55 = {};
    wp::int32 adj_56 = {};
    wp::float32 adj_57 = {};
    wp::float32 adj_58 = {};
    wp::int32 adj_59 = {};
    wp::float32 adj_60 = {};
    wp::float32 adj_61 = {};
    wp::float32 adj_62 = {};
    wp::float32 adj_63 = {};
    wp::float32 adj_64 = {};
    wp::float32 adj_65 = {};
    wp::float32 adj_66 = {};
    wp::float32 adj_67 = {};
    wp::float32 adj_68 = {};
    wp::mat_t<4, 4, wp::float32> adj_69 = {};
    //---------
    // forward
    // def transform_compose(position: vec3, rotation: quat, scale: vec3):                    <L 248>
    // R = wp.quat_to_matrix(rotation)                                                        <L 274>
    var_0 = wp::quat_to_matrix(var_rotation);
    // return mat44(                                                                          <L 276>
    // scale[0] * R[0,0], scale[1] * R[0,1], scale[2] * R[0,2], position[0],                  <L 277>
    var_2 = wp::extract(var_scale, var_1);
    var_5 = wp::extract(var_0, var_3, var_4);
    var_6 = wp::mul(var_2, var_5);
    var_8 = wp::extract(var_scale, var_7);
    var_11 = wp::extract(var_0, var_9, var_10);
    var_12 = wp::mul(var_8, var_11);
    var_14 = wp::extract(var_scale, var_13);
    var_17 = wp::extract(var_0, var_15, var_16);
    var_18 = wp::mul(var_14, var_17);
    var_20 = wp::extract(var_position, var_19);
    // scale[0] * R[1,0], scale[1] * R[1,1], scale[2] * R[1,2], position[1],                  <L 278>
    var_22 = wp::extract(var_scale, var_21);
    var_25 = wp::extract(var_0, var_23, var_24);
    var_26 = wp::mul(var_22, var_25);
    var_28 = wp::extract(var_scale, var_27);
    var_31 = wp::extract(var_0, var_29, var_30);
    var_32 = wp::mul(var_28, var_31);
    var_34 = wp::extract(var_scale, var_33);
    var_37 = wp::extract(var_0, var_35, var_36);
    var_38 = wp::mul(var_34, var_37);
    var_40 = wp::extract(var_position, var_39);
    // scale[0] * R[2,0], scale[1] * R[2,1], scale[2] * R[2,2], position[2],                  <L 279>
    var_42 = wp::extract(var_scale, var_41);
    var_45 = wp::extract(var_0, var_43, var_44);
    var_46 = wp::mul(var_42, var_45);
    var_48 = wp::extract(var_scale, var_47);
    var_51 = wp::extract(var_0, var_49, var_50);
    var_52 = wp::mul(var_48, var_51);
    var_54 = wp::extract(var_scale, var_53);
    var_57 = wp::extract(var_0, var_55, var_56);
    var_58 = wp::mul(var_54, var_57);
    var_60 = wp::extract(var_position, var_59);
    // dtype(0.0), dtype(0.0), dtype(0.0), dtype(1.0),                                        <L 280>
    var_62 = wp::float32(var_61);
    var_64 = wp::float32(var_63);
    var_66 = wp::float32(var_65);
    var_68 = wp::float32(var_67);
    var_69 = wp::mat_t<4, 4, wp::float32>(var_6, var_12, var_18, var_20, var_26, var_32, var_38, var_40, var_46, var_52, var_58, var_60, var_62, var_64, var_66, var_68);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_69 += adj_ret;
    wp::adj_mat_t(var_6, var_12, var_18, var_20, var_26, var_32, var_38, var_40, var_46, var_52, var_58, var_60, var_62, var_64, var_66, var_68, adj_6, adj_12, adj_18, adj_20, adj_26, adj_32, adj_38, adj_40, adj_46, adj_52, adj_58, adj_60, adj_62, adj_64, adj_66, adj_68, adj_69);
    wp::adj_float32(var_67, adj_67, adj_68);
    wp::adj_float32(var_65, adj_65, adj_66);
    wp::adj_float32(var_63, adj_63, adj_64);
    wp::adj_float32(var_61, adj_61, adj_62);
    // adj: dtype(0.0), dtype(0.0), dtype(0.0), dtype(1.0),                                   <L 280>
    wp::adj_extract(var_position, var_59, adj_position, adj_59, adj_60);
    wp::adj_mul(var_54, var_57, adj_54, adj_57, adj_58);
    wp::adj_extract(var_0, var_55, var_56, adj_0, adj_55, adj_56, adj_57);
    wp::adj_extract(var_scale, var_53, adj_scale, adj_53, adj_54);
    wp::adj_mul(var_48, var_51, adj_48, adj_51, adj_52);
    wp::adj_extract(var_0, var_49, var_50, adj_0, adj_49, adj_50, adj_51);
    wp::adj_extract(var_scale, var_47, adj_scale, adj_47, adj_48);
    wp::adj_mul(var_42, var_45, adj_42, adj_45, adj_46);
    wp::adj_extract(var_0, var_43, var_44, adj_0, adj_43, adj_44, adj_45);
    wp::adj_extract(var_scale, var_41, adj_scale, adj_41, adj_42);
    // adj: scale[0] * R[2,0], scale[1] * R[2,1], scale[2] * R[2,2], position[2],             <L 279>
    wp::adj_extract(var_position, var_39, adj_position, adj_39, adj_40);
    wp::adj_mul(var_34, var_37, adj_34, adj_37, adj_38);
    wp::adj_extract(var_0, var_35, var_36, adj_0, adj_35, adj_36, adj_37);
    wp::adj_extract(var_scale, var_33, adj_scale, adj_33, adj_34);
    wp::adj_mul(var_28, var_31, adj_28, adj_31, adj_32);
    wp::adj_extract(var_0, var_29, var_30, adj_0, adj_29, adj_30, adj_31);
    wp::adj_extract(var_scale, var_27, adj_scale, adj_27, adj_28);
    wp::adj_mul(var_22, var_25, adj_22, adj_25, adj_26);
    wp::adj_extract(var_0, var_23, var_24, adj_0, adj_23, adj_24, adj_25);
    wp::adj_extract(var_scale, var_21, adj_scale, adj_21, adj_22);
    // adj: scale[0] * R[1,0], scale[1] * R[1,1], scale[2] * R[1,2], position[1],             <L 278>
    wp::adj_extract(var_position, var_19, adj_position, adj_19, adj_20);
    wp::adj_mul(var_14, var_17, adj_14, adj_17, adj_18);
    wp::adj_extract(var_0, var_15, var_16, adj_0, adj_15, adj_16, adj_17);
    wp::adj_extract(var_scale, var_13, adj_scale, adj_13, adj_14);
    wp::adj_mul(var_8, var_11, adj_8, adj_11, adj_12);
    wp::adj_extract(var_0, var_9, var_10, adj_0, adj_9, adj_10, adj_11);
    wp::adj_extract(var_scale, var_7, adj_scale, adj_7, adj_8);
    wp::adj_mul(var_2, var_5, adj_2, adj_5, adj_6);
    wp::adj_extract(var_0, var_3, var_4, adj_0, adj_3, adj_4, adj_5);
    wp::adj_extract(var_scale, var_1, adj_scale, adj_1, adj_2);
    // adj: scale[0] * R[0,0], scale[1] * R[0,1], scale[2] * R[0,2], position[0],             <L 277>
    // adj: return mat44(                                                                     <L 276>
    wp::adj_quat_to_matrix(var_rotation, adj_rotation, adj_0);
    // adj: R = wp.quat_to_matrix(rotation)                                                   <L 274>
    // adj: def transform_compose(position: vec3, rotation: quat, scale: vec3):               <L 248>
    return;
}



extern "C" __global__ void decompose_fabric_transformation_matrix_to_warp_arrays_0f5b51e9_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::fabricarray_t<wp::mat_t<4, 4, wp::float64>> var_fabric_matrices,
    wp::array_t<wp::float32> var_array_positions,
    wp::array_t<wp::float32> var_array_orientations,
    wp::array_t<wp::float32> var_array_scales,
    wp::array_t<wp::uint32> var_indices,
    wp::array_t<wp::uint32> var_mapping)
{
    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        // reset shared memory allocator
        wp::tile_alloc_shared(0, true);

        //---------
        // primal vars
        wp::int32 var_0;
        wp::uint32* var_1;
        wp::uint32 var_2;
        wp::uint32 var_3;
        wp::uint32* var_4;
        wp::uint32 var_5;
        wp::uint32 var_6;
        wp::mat_t<4, 4, wp::float64>* var_7;
        wp::mat_t<4, 4, wp::float32> var_8;
        wp::mat_t<4, 4, wp::float64> var_9;
        wp::vec_t<3, wp::float32> var_10;
        wp::quat_t<wp::float32> var_11;
        wp::vec_t<3, wp::float32> var_12;
        const wp::int32 var_13 = 0;
        wp::shape_t* var_14;
        wp::int32 var_15;
        wp::shape_t var_16;
        const wp::int32 var_17 = 0;
        bool var_18;
        const wp::int32 var_19 = 0;
        wp::float32 var_20;
        const wp::int32 var_21 = 0;
        const wp::int32 var_22 = 1;
        wp::float32 var_23;
        const wp::int32 var_24 = 1;
        const wp::int32 var_25 = 2;
        wp::float32 var_26;
        const wp::int32 var_27 = 2;
        const wp::int32 var_28 = 0;
        wp::shape_t* var_29;
        wp::int32 var_30;
        wp::shape_t var_31;
        const wp::int32 var_32 = 0;
        bool var_33;
        const wp::int32 var_34 = 3;
        wp::float32 var_35;
        const wp::int32 var_36 = 0;
        const wp::int32 var_37 = 0;
        wp::float32 var_38;
        const wp::int32 var_39 = 1;
        const wp::int32 var_40 = 1;
        wp::float32 var_41;
        const wp::int32 var_42 = 2;
        const wp::int32 var_43 = 2;
        wp::float32 var_44;
        const wp::int32 var_45 = 3;
        const wp::int32 var_46 = 0;
        wp::shape_t* var_47;
        wp::int32 var_48;
        wp::shape_t var_49;
        const wp::int32 var_50 = 0;
        bool var_51;
        const wp::int32 var_52 = 0;
        wp::float32 var_53;
        const wp::int32 var_54 = 0;
        const wp::int32 var_55 = 1;
        wp::float32 var_56;
        const wp::int32 var_57 = 1;
        const wp::int32 var_58 = 2;
        wp::float32 var_59;
        const wp::int32 var_60 = 2;
        //---------
        // forward
        // def decompose_fabric_transformation_matrix_to_warp_arrays(                             <L 48>
        // output_index = wp.tid()                                                                <L 71>
        var_0 = builtin_tid1d();
        // view_index = indices[output_index]                                                     <L 73>
        var_1 = wp::address(var_indices, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // fabric_index = mapping[view_index]                                                     <L 75>
        var_4 = wp::address(var_mapping, var_2);
        var_6 = wp::load(var_4);
        var_5 = wp::copy(var_6);
        // position, rotation, scale = _decompose_transformation_matrix(wp.mat44f(fabric_matrices[fabric_index]))       <L 78>
        var_7 = wp::address(var_fabric_matrices, var_5);
        var_9 = wp::load(var_7);
        var_8 = wp::mat_t<4, 4, wp::float32>(var_9);
        _decompose_transformation_matrix_0(var_8, var_10, var_11, var_12);
        // if array_positions.shape[0] > 0:                                                       <L 80>
        var_14 = &(var_array_positions.shape);
        var_16 = wp::load(var_14);
        var_15 = wp::extract(var_16, var_13);
        var_18 = (var_15 > var_17);
        if (var_18) {
            // array_positions[output_index, 0] = position[0]                                     <L 81>
            var_20 = wp::extract(var_10, var_19);
            wp::array_store(var_array_positions, var_0, var_21, var_20);
            // array_positions[output_index, 1] = position[1]                                     <L 82>
            var_23 = wp::extract(var_10, var_22);
            wp::array_store(var_array_positions, var_0, var_24, var_23);
            // array_positions[output_index, 2] = position[2]                                     <L 83>
            var_26 = wp::extract(var_10, var_25);
            wp::array_store(var_array_positions, var_0, var_27, var_26);
        }
        // if array_orientations.shape[0] > 0:                                                    <L 85>
        var_29 = &(var_array_orientations.shape);
        var_31 = wp::load(var_29);
        var_30 = wp::extract(var_31, var_28);
        var_33 = (var_30 > var_32);
        if (var_33) {
            // array_orientations[output_index, 0] = rotation[3]  # w                             <L 86>
            var_35 = wp::extract(var_11, var_34);
            wp::array_store(var_array_orientations, var_0, var_36, var_35);
            // array_orientations[output_index, 1] = rotation[0]  # x                             <L 87>
            var_38 = wp::extract(var_11, var_37);
            wp::array_store(var_array_orientations, var_0, var_39, var_38);
            // array_orientations[output_index, 2] = rotation[1]  # y                             <L 88>
            var_41 = wp::extract(var_11, var_40);
            wp::array_store(var_array_orientations, var_0, var_42, var_41);
            // array_orientations[output_index, 3] = rotation[2]  # z                             <L 89>
            var_44 = wp::extract(var_11, var_43);
            wp::array_store(var_array_orientations, var_0, var_45, var_44);
        }
        // if array_scales.shape[0] > 0:                                                          <L 91>
        var_47 = &(var_array_scales.shape);
        var_49 = wp::load(var_47);
        var_48 = wp::extract(var_49, var_46);
        var_51 = (var_48 > var_50);
        if (var_51) {
            // array_scales[output_index, 0] = scale[0]                                           <L 92>
            var_53 = wp::extract(var_12, var_52);
            wp::array_store(var_array_scales, var_0, var_54, var_53);
            // array_scales[output_index, 1] = scale[1]                                           <L 93>
            var_56 = wp::extract(var_12, var_55);
            wp::array_store(var_array_scales, var_0, var_57, var_56);
            // array_scales[output_index, 2] = scale[2]                                           <L 94>
            var_59 = wp::extract(var_12, var_58);
            wp::array_store(var_array_scales, var_0, var_60, var_59);
        }
    }
}



extern "C" __global__ void compose_fabric_transformation_matrix_from_warp_arrays_206a7250_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::fabricarray_t<wp::mat_t<4, 4, wp::float64>> var_fabric_matrices,
    wp::array_t<wp::float32> var_array_positions,
    wp::array_t<wp::float32> var_array_orientations,
    wp::array_t<wp::float32> var_array_scales,
    bool var_broadcast_positions,
    bool var_broadcast_orientations,
    bool var_broadcast_scales,
    wp::array_t<wp::uint32> var_indices,
    wp::array_t<wp::uint32> var_mapping)
{
    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        // reset shared memory allocator
        wp::tile_alloc_shared(0, true);

        //---------
        // primal vars
        wp::int32 var_0;
        wp::uint32* var_1;
        wp::uint32* var_2;
        wp::uint32 var_3;
        wp::uint32 var_4;
        wp::uint32 var_5;
        wp::mat_t<4, 4, wp::float64>* var_6;
        wp::mat_t<4, 4, wp::float32> var_7;
        wp::mat_t<4, 4, wp::float64> var_8;
        wp::vec_t<3, wp::float32> var_9;
        wp::quat_t<wp::float32> var_10;
        wp::vec_t<3, wp::float32> var_11;
        const wp::int32 var_12 = 0;
        wp::shape_t* var_13;
        wp::int32 var_14;
        wp::shape_t var_15;
        const wp::int32 var_16 = 0;
        bool var_17;
        const wp::int32 var_18 = 0;
        wp::int32 var_19;
        wp::int32 var_20;
        const wp::int32 var_21 = 0;
        wp::float32* var_22;
        const wp::int32 var_23 = 0;
        wp::float32 var_24;
        const wp::int32 var_25 = 1;
        wp::float32* var_26;
        const wp::int32 var_27 = 1;
        wp::float32 var_28;
        const wp::int32 var_29 = 2;
        wp::float32* var_30;
        const wp::int32 var_31 = 2;
        wp::float32 var_32;
        const wp::int32 var_33 = 0;
        wp::shape_t* var_34;
        wp::int32 var_35;
        wp::shape_t var_36;
        const wp::int32 var_37 = 0;
        bool var_38;
        const wp::int32 var_39 = 0;
        wp::int32 var_40;
        wp::int32 var_41;
        wp::int32 var_42;
        const wp::int32 var_43 = 1;
        wp::float32* var_44;
        const wp::int32 var_45 = 0;
        wp::float32 var_46;
        const wp::int32 var_47 = 2;
        wp::float32* var_48;
        const wp::int32 var_49 = 1;
        wp::float32 var_50;
        const wp::int32 var_51 = 3;
        wp::float32* var_52;
        const wp::int32 var_53 = 2;
        wp::float32 var_54;
        const wp::int32 var_55 = 0;
        wp::float32* var_56;
        const wp::int32 var_57 = 3;
        wp::float32 var_58;
        wp::int32 var_59;
        const wp::int32 var_60 = 0;
        wp::shape_t* var_61;
        wp::int32 var_62;
        wp::shape_t var_63;
        const wp::int32 var_64 = 0;
        bool var_65;
        const wp::int32 var_66 = 0;
        wp::int32 var_67;
        wp::int32 var_68;
        wp::int32 var_69;
        const wp::int32 var_70 = 0;
        wp::float32* var_71;
        const wp::int32 var_72 = 0;
        wp::float32 var_73;
        const wp::int32 var_74 = 1;
        wp::float32* var_75;
        const wp::int32 var_76 = 1;
        wp::float32 var_77;
        const wp::int32 var_78 = 2;
        wp::float32* var_79;
        const wp::int32 var_80 = 2;
        wp::float32 var_81;
        wp::int32 var_82;
        wp::mat_t<4, 4, wp::float32> var_83;
        wp::mat_t<4, 4, wp::float32> var_84;
        wp::mat_t<4, 4, wp::float64> var_85;
        //---------
        // forward
        // def compose_fabric_transformation_matrix_from_warp_arrays(                             <L 98>
        // i = wp.tid()                                                                           <L 128>
        var_0 = builtin_tid1d();
        // fabric_index = mapping[indices[i]]                                                     <L 130>
        var_1 = wp::address(var_indices, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::address(var_mapping, var_3);
        var_5 = wp::load(var_2);
        var_4 = wp::copy(var_5);
        // position, rotation, scale = _decompose_transformation_matrix(wp.mat44f(fabric_matrices[fabric_index]))       <L 132>
        var_6 = wp::address(var_fabric_matrices, var_4);
        var_8 = wp::load(var_6);
        var_7 = wp::mat_t<4, 4, wp::float32>(var_8);
        _decompose_transformation_matrix_0(var_7, var_9, var_10, var_11);
        // if array_positions.shape[0] > 0:                                                       <L 134>
        var_13 = &(var_array_positions.shape);
        var_15 = wp::load(var_13);
        var_14 = wp::extract(var_15, var_12);
        var_17 = (var_14 > var_16);
        if (var_17) {
            // if broadcast_positions:                                                            <L 135>
            if (var_broadcast_positions) {
                // index = 0                                                                      <L 136>
            }
            if (!var_broadcast_positions) {
                // index = i                                                                      <L 138>
                var_19 = wp::copy(var_0);
            }
            var_20 = wp::where(var_broadcast_positions, var_18, var_19);
            // position[0] = array_positions[index, 0]                                            <L 139>
            var_22 = wp::address(var_array_positions, var_20, var_21);
            var_24 = wp::load(var_22);
            wp::assign_inplace(var_9, var_23, var_24);
            // position[1] = array_positions[index, 1]                                            <L 140>
            var_26 = wp::address(var_array_positions, var_20, var_25);
            var_28 = wp::load(var_26);
            wp::assign_inplace(var_9, var_27, var_28);
            // position[2] = array_positions[index, 2]                                            <L 141>
            var_30 = wp::address(var_array_positions, var_20, var_29);
            var_32 = wp::load(var_30);
            wp::assign_inplace(var_9, var_31, var_32);
        }
        // if array_orientations.shape[0] > 0:                                                    <L 143>
        var_34 = &(var_array_orientations.shape);
        var_36 = wp::load(var_34);
        var_35 = wp::extract(var_36, var_33);
        var_38 = (var_35 > var_37);
        if (var_38) {
            // if broadcast_orientations:                                                         <L 144>
            if (var_broadcast_orientations) {
                // index = 0                                                                      <L 145>
            }
            var_40 = wp::where(var_broadcast_orientations, var_39, var_20);
            if (!var_broadcast_orientations) {
                // index = i                                                                      <L 147>
                var_41 = wp::copy(var_0);
            }
            var_42 = wp::where(var_broadcast_orientations, var_40, var_41);
            // rotation[0] = array_orientations[index, 1]  # x                                    <L 148>
            var_44 = wp::address(var_array_orientations, var_42, var_43);
            var_46 = wp::load(var_44);
            wp::assign_inplace(var_10, var_45, var_46);
            // rotation[1] = array_orientations[index, 2]  # y                                    <L 149>
            var_48 = wp::address(var_array_orientations, var_42, var_47);
            var_50 = wp::load(var_48);
            wp::assign_inplace(var_10, var_49, var_50);
            // rotation[2] = array_orientations[index, 3]  # z                                    <L 150>
            var_52 = wp::address(var_array_orientations, var_42, var_51);
            var_54 = wp::load(var_52);
            wp::assign_inplace(var_10, var_53, var_54);
            // rotation[3] = array_orientations[index, 0]  # w                                    <L 151>
            var_56 = wp::address(var_array_orientations, var_42, var_55);
            var_58 = wp::load(var_56);
            wp::assign_inplace(var_10, var_57, var_58);
        }
        var_59 = wp::where(var_38, var_42, var_20);
        // if array_scales.shape[0] > 0:                                                          <L 153>
        var_61 = &(var_array_scales.shape);
        var_63 = wp::load(var_61);
        var_62 = wp::extract(var_63, var_60);
        var_65 = (var_62 > var_64);
        if (var_65) {
            // if broadcast_scales:                                                               <L 154>
            if (var_broadcast_scales) {
                // index = 0                                                                      <L 155>
            }
            var_67 = wp::where(var_broadcast_scales, var_66, var_59);
            if (!var_broadcast_scales) {
                // index = i                                                                      <L 157>
                var_68 = wp::copy(var_0);
            }
            var_69 = wp::where(var_broadcast_scales, var_67, var_68);
            // scale[0] = array_scales[index, 0]                                                  <L 158>
            var_71 = wp::address(var_array_scales, var_69, var_70);
            var_73 = wp::load(var_71);
            wp::assign_inplace(var_11, var_72, var_73);
            // scale[1] = array_scales[index, 1]                                                  <L 159>
            var_75 = wp::address(var_array_scales, var_69, var_74);
            var_77 = wp::load(var_75);
            wp::assign_inplace(var_11, var_76, var_77);
            // scale[2] = array_scales[index, 2]                                                  <L 160>
            var_79 = wp::address(var_array_scales, var_69, var_78);
            var_81 = wp::load(var_79);
            wp::assign_inplace(var_11, var_80, var_81);
        }
        var_82 = wp::where(var_65, var_69, var_59);
        // fabric_matrices[fabric_index] = wp.mat44d(  # type: ignore[arg-type]                   <L 163>
        // wp.transpose(wp.transform_compose(position, rotation, scale))  # type: ignore[arg-type]       <L 164>
        var_83 = transform_compose_0(var_9, var_10, var_11);
        var_84 = wp::transpose(var_83);
        var_85 = wp::mat_t<4, 4, wp::float64>(var_84);
        // fabric_matrices[fabric_index] = wp.mat44d(  # type: ignore[arg-type]                   <L 163>
        wp::array_store(var_fabric_matrices, var_4, var_85);
    }
}



extern "C" __global__ void set_view_to_fabric_array_f0d1eac8_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::fabricarray_t<wp::uint32> var_fabric_to_view,
    wp::array_t<wp::uint32> var_view_to_fabric)
{
    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        // reset shared memory allocator
        wp::tile_alloc_shared(0, true);

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::uint32* var_2;
        wp::int32 var_3;
        wp::uint32 var_4;
        wp::uint32 var_5;
        //---------
        // forward
        // def set_view_to_fabric_array(fabric_to_view: FabricArrayUInt32, view_to_fabric: ArrayUInt32):       <L 33>
        // fabric_idx = int(wp.tid())                                                             <L 35>
        var_0 = builtin_tid1d();
        var_1 = wp::int(var_0);
        // view_idx = int(fabric_to_view[fabric_idx])                                             <L 36>
        var_2 = wp::address(var_fabric_to_view, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::int(var_4);
        // view_to_fabric[view_idx] = wp.uint32(fabric_idx)                                       <L 37>
        var_5 = wp::uint32(var_1);
        wp::array_store(var_view_to_fabric, var_3, var_5);
    }
}



extern "C" __global__ void arange_k_7ed6bcda_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::uint32> var_a)
{
    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        // reset shared memory allocator
        wp::tile_alloc_shared(0, true);

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::uint32 var_2;
        //---------
        // forward
        // def arange_k(a: ArrayUInt32_1d):                                                       <L 41>
        // tid = int(wp.tid())                                                                    <L 43>
        var_0 = builtin_tid1d();
        var_1 = wp::int(var_0);
        // a[tid] = wp.uint32(tid)                                                                <L 44>
        var_2 = wp::uint32(var_1);
        wp::array_store(var_a, var_1, var_2);
    }
}

