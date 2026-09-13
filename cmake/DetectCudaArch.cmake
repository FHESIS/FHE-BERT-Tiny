# Auto-detect the CUDA architecture of the GPU(s) present and default FIDESLIB_ARCH to just
# that target, instead of FIDESlib's own default 7-architecture portability matrix
# (80/86/89/90/100/120). That matrix means ~7x the CUDA device compile + device-link work for
# archs nothing in this environment uses. Pass -DFIDESLIB_ARCH=... explicitly (e.g. for a
# release build meant to run on GPUs other than the one it's built on) to override this.
if (NOT DEFINED FIDESLIB_ARCH)
    find_program(NVIDIA_SMI_EXECUTABLE nvidia-smi)
    if (NVIDIA_SMI_EXECUTABLE)
        execute_process(
                COMMAND "${NVIDIA_SMI_EXECUTABLE}" --query-gpu=compute_cap --format=csv,noheader
                OUTPUT_VARIABLE _detected_compute_caps
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
                RESULT_VARIABLE _nvidia_smi_result
        )
        if (_nvidia_smi_result EQUAL 0 AND _detected_compute_caps)
            string(REPLACE "\n" ";" _compute_cap_list "${_detected_compute_caps}")
            list(REMOVE_DUPLICATES _compute_cap_list)
            list(GET _compute_cap_list 0 _compute_cap)
            string(REPLACE "." "" _cuda_arch "${_compute_cap}")
            set(FIDESLIB_ARCH "${_cuda_arch}" CACHE STRING "Set the backend architecture target.")
            message(STATUS "DetectCudaArch: detected GPU compute capability ${_compute_cap} -> FIDESLIB_ARCH=${FIDESLIB_ARCH}")
        else ()
            message(STATUS "DetectCudaArch: nvidia-smi query failed; leaving FIDESLIB_ARCH at FIDESlib's default")
        endif ()
    else ()
        message(STATUS "DetectCudaArch: nvidia-smi not found; leaving FIDESLIB_ARCH at FIDESlib's default")
    endif ()
endif ()
