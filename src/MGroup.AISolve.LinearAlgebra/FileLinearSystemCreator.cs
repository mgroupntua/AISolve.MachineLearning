namespace MGroup.AISolve.LinearAlgebra
{
	using System;
	using System.Collections.Generic;
	using System.IO;
	using System.Linq;
	using System.Text;
	using System.Threading.Tasks;

	using MGroup.LinearAlgebra.Matrices;
	using MGroup.LinearAlgebra.Matrices.Builders;
	using MGroup.LinearAlgebra.Vectors;

	public class FileLinearSystemCreator : ILinearSystemCreator
	{
		private double noise;
		private double rhsRandomness;
		private Random random = new Random();
		
		public CsrMatrix A { get; }
		public Vector RHS { get; }

		public FileLinearSystemCreator(string matrixFilename, string rhsFilename, double noise, double rhsRandomness) 
		{
			this.noise = noise;
			this.rhsRandomness = rhsRandomness;

			var lines = File.ReadAllLines(matrixFilename).Where(x => x.StartsWith("%") == false).ToArray();
			var lineValues = lines[0].Split(' ').Select(x => Int32.Parse(x)).ToArray();
			var rows = new int[lineValues[0]];
			var cols = new int[lineValues[1]];
			var values = new double[lineValues[2]];
			var lineEntries = lines.Skip(1).Select(x => (Int32.Parse(x.Split(' ', StringSplitOptions.RemoveEmptyEntries)[1]) - 1,
				Int32.Parse(x.Split(' ', StringSplitOptions.RemoveEmptyEntries)[0]) - 1, Double.Parse(x.Split(' ', StringSplitOptions.RemoveEmptyEntries)[2])));
			var matrix = DokSymmetric.CreateFromSparsePattern(lineValues[0], lineEntries);
			A = CsrMatrix.CreateFromDense(matrix);

			if (String.IsNullOrWhiteSpace(rhsFilename) == false) 
			{
				var rhsLines = File.ReadAllLines(rhsFilename).Where(x => x.StartsWith("%") == false).ToArray();
				var rhsValues = rhsLines.Select(x => Double.Parse(x)).ToArray();
				RHS = Vector.CreateFromArray(rhsValues);
			}
			else
			{
				RHS = Vector.CreateZero(matrix.NumRows);
				for (int i = 0; i < matrix.NumRows / 10; i++)
				{
					RHS[i * 10] = 1e-5 * random.NextDouble();
				}
			}
		}

		public (CsrMatrix, Vector) GetModel(double[] parameterValues)
		{
			var factor = parameterValues[0];
			var m = A.Copy(false);
			for (int i = 0; i < m.RawValues.Length; i++) 
			{
				m.RawValues[i] *= 1 + factor * (1 + noise * (random.NextDouble() - 0.5d));
			}

			var r = (Vector)RHS.Copy(false);
			for (int i = 0; i < r.Length; i++)
			{
				r[i] *= 1 + rhsRandomness * (random.NextDouble() - 0.5d);
			}

			return (m, r);
		}
	}
}
