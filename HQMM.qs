namespace HQMM {
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Math;
    open Microsoft.Quantum.Convert;
    open Microsoft.Quantum.Arrays;
    open Microsoft.Quantum.Synthesis;
    open Microsoft.Quantum.Arithmetic;

    function DoubleToComplexMatrix(m : Double[][]) : Microsoft.Quantum.Math.Complex[][] {
        let rows = Length(ColumnAt(0,m));
        mutable cmatrix = [
            [Complex(m[0][0],0.0),Complex(m[0][1],0.0),Complex(m[0][2],0.0),Complex(m[0][3],0.0)],
            [Complex(m[1][0],0.0),Complex(m[1][1],0.0),Complex(m[1][2],0.0),Complex(m[1][3],0.0)],
            [Complex(m[2][0],0.0),Complex(m[2][1],0.0),Complex(m[2][2],0.0),Complex(m[2][3],0.0)],
            [Complex(m[3][0],0.0),Complex(m[3][1],0.0),Complex(m[3][2],0.0),Complex(m[3][3],0.0)]
        ];
        return cmatrix;
    }

    operation StartExperiment(transmat : Double[][],
                              startprob : Double[][],
                              emissionprob : Double[][]) : Int {
        use q = Qubit[2];
        ApplyToEach(H, q);
        let register = LittleEndian(q);
        ApplyUnitary(DoubleToComplexMatrix(transmat),register);
        ApplyUnitary(DoubleToComplexMatrix(startprob),register);
        ApplyUnitary(DoubleToComplexMatrix(emissionprob),register);
        let res = MeasureInteger(register);
        return res;
    }
}