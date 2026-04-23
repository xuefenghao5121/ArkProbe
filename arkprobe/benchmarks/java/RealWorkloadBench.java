/**
 * ArkProbe Real Workload Benchmark — compute-intensive Java applications
 * with C++ acceleration via JNI.
 *
 * Workloads (real algorithms, not micro-benchmarks):
 *   1. FFT — Cooley-Tukey radix-2 1D FFT (signal processing)
 *   2. SOR — Red-black Successive Over-Relaxation (PDE solver)
 *   3. SparseMatvec — CSR sparse matrix-vector multiply (iterative solver)
 *   4. LU — LU decomposition with partial pivoting (linear algebra)
 *   5. KMeans — Lloyd's algorithm assignment step (ML preprocessing)
 *
 * Usage:
 *   java RealWorkloadBench                          — pure Java baseline
 *   java -Djava.library.path=<dir> RealWorkloadBench  — with C++ acceleration
 *
 * Output: JSON with timing per workload.
 */
import java.util.*;

public class RealWorkloadBench {

    // ========== native method declarations (C++ .so provides these) ==========
    public static native void fftTransform(double[] re, double[] im, int n, boolean inverse);
    public static native void sorIteration(double[] grid, int rows, int cols,
                                           double omega, int iters);
    public static native void sparseMatvec(double[] val, int[] rowPtr, int[] colIdx,
                                           double[] x, double[] y, int nrows);
    public static native void luDecompose(double[] a, int n, int[] pivot);
    public static native void kmeansAssign(double[] data, double[] centroids,
                                           int[] assignments, int n, int k, int dim);

    // ========== Java implementations ==========

    /** Cooley-Tukey radix-2 in-place FFT. n must be power of 2. */
    public static void fftJava(double[] re, double[] im, int n, boolean inverse) {
        // bit-reversal permutation
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            while ((j & bit) != 0) { j ^= bit; bit >>= 1; }
            j ^= bit;
            if (i < j) {
                double tmp = re[i]; re[i] = re[j]; re[j] = tmp;
                tmp = im[i]; im[i] = im[j]; im[j] = tmp;
            }
        }
        // butterfly stages
        double sign = inverse ? 1.0 : -1.0;
        for (int len = 2; len <= n; len <<= 1) {
            double ang = 2.0 * Math.PI / len * sign;
            double wRe = Math.cos(ang);
            double wIm = Math.sin(ang);
            for (int i = 0; i < n; i += len) {
                double curRe = 1.0, curIm = 0.0;
                for (int j = 0; j < len / 2; j++) {
                    int u = i + j;
                    int v = i + j + len / 2;
                    double tRe = curRe * re[v] - curIm * im[v];
                    double tIm = curRe * im[v] + curIm * re[v];
                    re[v] = re[u] - tRe;
                    im[v] = im[u] - tIm;
                    re[u] += tRe;
                    im[u] += tIm;
                    double newCurRe = curRe * wRe - curIm * wIm;
                    curIm = curRe * wIm + curIm * wRe;
                    curRe = newCurRe;
                }
            }
        }
        if (inverse) {
            for (int i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
        }
    }

    /** Red-black SOR iteration on a flat row-major grid. */
    public static void sorJava(double[] grid, int rows, int cols,
                               double omega, int iters) {
        for (int it = 0; it < iters; it++) {
            for (int parity = 0; parity <= 1; parity++) {
                for (int i = 1; i < rows - 1; i++) {
                    int startJ = ((i + parity) % 2) + 1;
                    for (int j = startJ; j < cols - 1; j += 2) {
                        int idx = i * cols + j;
                        double newVal = 0.25 * (grid[idx - cols] + grid[idx + cols] +
                                grid[idx - 1] + grid[idx + 1]);
                        grid[idx] = (1.0 - omega) * grid[idx] + omega * newVal;
                    }
                }
            }
        }
    }

    /** CSR sparse matrix-vector multiply: y = A * x. */
    public static void sparseMatvecJava(double[] val, int[] rowPtr, int[] colIdx,
                                        double[] x, double[] y, int nrows) {
        for (int i = 0; i < nrows; i++) {
            double sum = 0.0;
            for (int j = rowPtr[i]; j < rowPtr[i + 1]; j++) {
                sum += val[j] * x[colIdx[j]];
            }
            y[i] = sum;
        }
    }

    /** LU decomposition with partial pivoting. a is n×n in row-major. */
    public static void luJava(double[] a, int n, int[] pivot) {
        for (int i = 0; i < n; i++) pivot[i] = i;
        for (int k = 0; k < n - 1; k++) {
            // find pivot
            int maxRow = k;
            double maxVal = Math.abs(a[k * n + k]);
            for (int i = k + 1; i < n; i++) {
                double v = Math.abs(a[i * n + k]);
                if (v > maxVal) { maxVal = v; maxRow = i; }
            }
            // swap rows
            if (maxRow != k) {
                for (int j = 0; j < n; j++) {
                    int aIdx = k * n + j, bIdx = maxRow * n + j;
                    double tmp = a[aIdx]; a[aIdx] = a[bIdx]; a[bIdx] = tmp;
                }
                int tmp = pivot[k]; pivot[k] = pivot[maxRow]; pivot[maxRow] = tmp;
            }
            // eliminate
            double diag = a[k * n + k];
            if (Math.abs(diag) < 1e-15) continue;
            for (int i = k + 1; i < n; i++) {
                double factor = a[i * n + k] / diag;
                a[i * n + k] = factor;
                for (int j = k + 1; j < n; j++) {
                    a[i * n + j] -= factor * a[k * n + j];
                }
            }
        }
    }

    /** KMeans assignment: for each point, find nearest centroid. */
    public static void kmeansJava(double[] data, double[] centroids,
                                  int[] assignments, int n, int k, int dim) {
        for (int i = 0; i < n; i++) {
            double minDist = Double.MAX_VALUE;
            int best = 0;
            for (int c = 0; c < k; c++) {
                double dist = 0.0;
                for (int d = 0; d < dim; d++) {
                    double diff = data[i * dim + d] - centroids[c * dim + d];
                    dist += diff * diff;
                }
                if (dist < minDist) { minDist = dist; best = c; }
            }
            assignments[i] = best;
        }
    }

    // ========== benchmark harness ==========

    private static final int WARMUP_ITERS = 5;

    static long measure(Runnable r, int iterations) {
        for (int i = 0; i < WARMUP_ITERS; i++) r.run();
        long start = System.nanoTime();
        for (int i = 0; i < iterations; i++) r.run();
        return System.nanoTime() - start;
    }

    // ========== data generators ==========

    static double[] generateSignal(int n, Random rng) {
        double[] re = new double[n];
        double[] im = new double[n];
        for (int i = 0; i < n; i++) {
            re[i] = Math.sin(2 * Math.PI * 3 * i / n) +
                    0.5 * Math.sin(2 * Math.PI * 7 * i / n) +
                    rng.nextGaussian() * 0.1;
            im[i] = 0.0;
        }
        return re;
    }

    static double[] generateGrid(int rows, int cols, Random rng) {
        double[] grid = new double[rows * cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++) {
                double v = 0.0;
                if (i == 0 || i == rows - 1) v = 1.0;
                else if (j == 0 || j == cols - 1) v = 1.0;
                else v = rng.nextDouble() * 0.01;
                grid[i * cols + j] = v;
            }
        return grid;
    }

    static Object[] generateSparse(int nrows, int ncols, int nnzPerRow, Random rng) {
        int nnz = nrows * nnzPerRow;
        double[] val = new double[nnz];
        int[] colIdx = new int[nnz];
        int[] rowPtr = new int[nrows + 1];
        int idx = 0;
        for (int i = 0; i < nrows; i++) {
            rowPtr[i] = idx;
            Set<Integer> cols = new TreeSet<>();
            while (cols.size() < nnzPerRow) cols.add(rng.nextInt(ncols));
            for (int c : cols) {
                val[idx] = rng.nextDouble();
                colIdx[idx] = c;
                idx++;
            }
        }
        rowPtr[nrows] = nnz;
        double[] x = new double[ncols];
        for (int i = 0; i < ncols; i++) x[i] = rng.nextDouble();
        return new Object[]{val, rowPtr, colIdx, x};
    }

    static double[] generateMatrix(int n, Random rng) {
        double[] a = new double[n * n];
        for (int i = 0; i < n * n; i++) a[i] = rng.nextDouble();
        return a;
    }

    static double[] generateKMeansData(int n, int dim, int k, Random rng) {
        double[] data = new double[n * dim];
        double[] centroids = new double[k * dim];
        for (int c = 0; c < k; c++)
            for (int d = 0; d < dim; d++)
                centroids[c * dim + d] = rng.nextDouble() * 100.0;
        for (int i = 0; i < n; i++) {
            int center = rng.nextInt(k);
            for (int d = 0; d < dim; d++)
                data[i * dim + d] = centroids[center * dim + d] + rng.nextGaussian() * 2.0;
        }
        return data;
    }

    static double[] generateCentroids(int k, int dim, Random rng) {
        double[] centroids = new double[k * dim];
        for (int i = 0; i < k * dim; i++)
            centroids[i] = rng.nextDouble() * 100.0;
        return centroids;
    }

    // ========== main ==========

    public static void main(String[] args) throws Exception {
        boolean nativeLoaded = false;
        try {
            System.loadLibrary("arkprobe_realworkload");
            nativeLoaded = true;
            System.err.println("[INFO] Native library loaded.");
        } catch (UnsatisfiedLinkError e) {
            System.err.println("[INFO] Native library not found, Java-only mode.");
        }

        Random rng = new Random(42);

        // ---- data sizes ----
        int fftN = 1 << 20;      // 1M points
        int sorRows = 500;
        int sorCols = 500;
        int sorIters = 50;
        double sorOmega = 1.5;
        int sparseRows = 100000;
        int sparseCols = 100000;
        int nnzPerRow = 10;
        int luN = 512;
        int kmeansN = 100000;
        int kmeansK = 64;
        int kmeansDim = 32;

        // ---- generate data ----
        double[] fftRe = generateSignal(fftN, rng);
        double[] fftIm = new double[fftN];
        double[] fftImCopy = new double[fftN]; // for re-running

        double[] grid = generateGrid(sorRows, sorCols, rng);
        double[] gridCopy = grid.clone();

        Object[] sparse = generateSparse(sparseRows, sparseCols, nnzPerRow, rng);
        double[] spVal = (double[]) sparse[0];
        int[] spRowPtr = (int[]) sparse[1];
        int[] spColIdx = (int[]) sparse[2];
        double[] spX = (double[]) sparse[3];
        double[] spY = new double[sparseRows];

        double[] luA = generateMatrix(luN, rng);
        int[] luPivot = new int[luN];

        double[] kmData = generateKMeansData(kmeansN, kmeansDim, kmeansK, rng);
        double[] kmCentroids = generateCentroids(kmeansK, kmeansDim, rng);
        int[] kmAssign = new int[kmeansN];

        // ---- iteration counts ----
        int fftIters = 50;
        int sorItersBench = 10;
        int sparseIters = 200;
        int luIters = 10;
        int kmIters = 50;

        StringBuilder json = new StringBuilder();
        json.append("{\n");

        // ========== FFT ==========
        double[] fftRe1 = fftRe.clone();
        double[] fftIm1 = fftIm.clone();
        appendResult(json, "fft_java",
            measure(() -> fftJava(fftRe1, fftIm1, fftN, false), fftIters));
        if (nativeLoaded) {
            double[] fftRe2 = fftRe.clone();
            double[] fftIm2 = fftIm.clone();
            appendResult(json, "fft_cpp",
                measure(() -> fftTransform(fftRe2, fftIm2, fftN, false), fftIters));
        }

        // ========== SOR ==========
        double[] grid1 = grid.clone();
        appendResult(json, "sor_java",
            measure(() -> sorJava(grid1, sorRows, sorCols, sorOmega, sorIters), sorItersBench));
        if (nativeLoaded) {
            double[] grid2 = grid.clone();
            appendResult(json, "sor_cpp",
                measure(() -> sorIteration(grid2, sorRows, sorCols, sorOmega, sorIters), sorItersBench));
        }

        // ========== Sparse Matvec ==========
        double[] spY1 = new double[sparseRows];
        appendResult(json, "sparse_matvec_java",
            measure(() -> {
                Arrays.fill(spY1, 0.0);
                sparseMatvecJava(spVal, spRowPtr, spColIdx, spX, spY1, sparseRows);
            }, sparseIters));
        if (nativeLoaded) {
            double[] spY2 = new double[sparseRows];
            appendResult(json, "sparse_matvec_cpp",
                measure(() -> {
                    Arrays.fill(spY2, 0.0);
                    sparseMatvec(spVal, spRowPtr, spColIdx, spX, spY2, sparseRows);
                }, sparseIters));
        }

        // ========== LU Decomposition ==========
        double[] luA1 = luA.clone();
        int[] piv1 = new int[luN];
        appendResult(json, "lu_java",
            measure(() -> luJava(luA1.clone(), luN, piv1), luIters));
        if (nativeLoaded) {
            double[] luA2 = luA.clone();
            int[] piv2 = new int[luN];
            appendResult(json, "lu_cpp",
                measure(() -> luDecompose(luA2.clone(), luN, piv2), luIters));
        }

        // ========== KMeans Assignment ==========
        double[] kmData1 = kmData.clone();
        double[] kmCent1 = kmCentroids.clone();
        int[] kmAssn1 = new int[kmeansN];
        appendResult(json, "kmeans_java",
            measure(() -> kmeansJava(kmData1, kmCent1, kmAssn1, kmeansN, kmeansK, kmeansDim), kmIters));
        if (nativeLoaded) {
            double[] kmData2 = kmData.clone();
            double[] kmCent2 = kmCentroids.clone();
            int[] kmAssn2 = new int[kmeansN];
            appendResult(json, "kmeans_cpp",
                measure(() -> kmeansAssign(kmData2, kmCent2, kmAssn2, kmeansN, kmeansK, kmeansDim), kmIters));
        }

        // remove trailing comma
        if (json.length() > 2 && json.charAt(json.length() - 2) == ',') {
            json.deleteCharAt(json.length() - 2);
        }
        json.append("}\n");

        System.out.println(json.toString());
    }

    private static void appendResult(StringBuilder json, String key, long nanos) {
        double ms = nanos / 1_000_000.0;
        json.append(String.format("  \"%s_ms\": %.3f,\n", key, ms));
    }
}
