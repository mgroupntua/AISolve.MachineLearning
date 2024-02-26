using MGroup.LinearAlgebra.Matrices;
using MGroup.LinearAlgebra.Vectors;

namespace MGroup.AISolve.LinearAlgebra
{
    /// <summary>
    /// Defines the generation of a linear system with specific parameters.
    /// </summary>
    public interface ILinearSystemCreator
    {
		/// <summary>
		/// Generates a matrix of coefficients and a right hand side that takes into account the <paramref name="parameterValues"/> provided.
		/// </summary>
		/// <param name="parameterValues">A double array containing arbitrary parameters ordered as in <see cref="IModelResponse.GetModelResponse"/>.</param>
		/// <returns>A tuple of a <see cref="CsrMatrix"/> and a <see cref="Vector"/>, made using the <paramref name="parameterValues"/> provided.</returns>
		(CsrMatrix, Vector) GetModel(double[] parameterValues);
    }
}
