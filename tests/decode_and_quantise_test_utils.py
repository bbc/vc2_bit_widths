from vc2_conformance.pseudocode.state import State

from vc2_conformance.pseudocode.picture_encoding import forward_wavelet_transform

from vc2_conformance.pseudocode.picture_decoding import inverse_wavelet_transform

from vc2_bit_widths.quantisation import forward_quant, inverse_quant


def encode_with_vc2(picture, width, height,
                    wavelet_index, wavelet_index_ho, dwt_depth, dwt_depth_ho):
    state = State(
        luma_width=width,
        luma_height=height,
        color_diff_width=0,
        color_diff_height=0,
        
        wavelet_index=wavelet_index,
        wavelet_index_ho=wavelet_index_ho,
        dwt_depth=dwt_depth,
        dwt_depth_ho=dwt_depth_ho,
    )
    
    current_picture = {
        "Y": picture,
        "C1": [],
        "C2": [],
    }
    
    forward_wavelet_transform(state, current_picture)
    
    return state["y_transform"]


def quantise_coeffs(transform_coeffs, qi, quantisation_matrix={}):
    return {
        level: {
            orient: [
                [
                    inverse_quant(
                        forward_quant(
                            value,
                            max(0, qi - quantisation_matrix.get(level, {}).get(orient, 0)),
                        ),
                        max(0, qi - quantisation_matrix.get(level, {}).get(orient, 0)),
                    )
                    for value in row
                ]
                for row in array
            ]
            for orient, array in orients.items()
        }
        for level, orients in transform_coeffs.items()
    }


def decode_with_vc2(transform_coeffs, width, height,
                    wavelet_index, wavelet_index_ho, dwt_depth, dwt_depth_ho):
    state = State(
        luma_width=width,
        luma_height=height,
        color_diff_width=0,
        color_diff_height=0,
        
        wavelet_index=wavelet_index,
        wavelet_index_ho=wavelet_index_ho,
        dwt_depth=dwt_depth,
        dwt_depth_ho=dwt_depth_ho,
        
        y_transform=transform_coeffs,
        c1_transform=transform_coeffs,  # Ignored
        c2_transform=transform_coeffs,  # Ignored
        
        current_picture={},
    )
    
    inverse_wavelet_transform(state)
    
    return state["current_picture"]["Y"]
