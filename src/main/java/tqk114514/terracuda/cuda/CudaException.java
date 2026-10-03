package tqk114514.terracuda.cuda;

/**
 * Raised when a CUDA Driver API call returns a {@code CUresult} other than {@code CUDA_SUCCESS},
 * or when the FFM binding itself cannot be invoked.
 *
 * <p>The mod must never let a GPU failure crash the JVM: callers are expected to catch this and fall
 * back to the vanilla code path (see the degradation matrix in the design doc).
 */
public class CudaException extends RuntimeException {

    private static final long serialVersionUID = 1L;

    /** Raw {@code CUresult} code, or {@code -1} when the failure happened inside the FFM binding. */
    public static final int BINDING_FAILURE = -1;

    private final int code;
    private final String errorName;

    public CudaException(int code, String errorName, String message) {
        super(message);
        this.code = code;
        this.errorName = errorName;
    }

    public CudaException(int code, String errorName, String message, Throwable cause) {
        super(message, cause);
        this.code = code;
        this.errorName = errorName;
    }

    public int code() {
        return code;
    }

    public String errorName() {
        return errorName;
    }

    @Override
    public String toString() {
        return "CudaException[" + errorName + " (" + code + ")]: " + getMessage();
    }
}
