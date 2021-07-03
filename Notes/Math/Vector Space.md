A vector space $\mathbb{R}^n$ is spanned by a set of linearly independent vectors $\mathcal{X} = \{x_1,\cdots,x_k\}$ i.e. $$\sum_{i = 1}^k a_i x_i = \mathbf{0}$$ has the only solution that $x_i = \mathbf{0}$. $\mathcal{X}$ is called the **basis** of $\mathbb{R}^n$, and $k$ is the dimension of vector space $\mathbb{R}^n$.

Let's consider the example of $Ax = \mathbf{b}$, where $A\in \mathbb{R}^{m\times n}$. The solution of the augmented equation system $(A\mid \mathbf{b})$ is calculated as:
- find a particular solution to $Ax = \mathbf{b}$
- find all solutions to $Ax = \mathbf{0}$
- combine (add) the solutions from the last two steps

In order to find all solutions to $Ax = \mathbf{0}$, we have to:
- using **Gaussian Elimination** to transform the augmented matrix $(A\mid \mathbf{0})$ to the **Reduced Row Echelon Form**
- express the **non-pivot** the column by the linear combination of columns on their left, obtaining a vector $\lambda_i$
- repeat last step until all non-pivot columns are processed, leading to a set of vectors $\mathcal{O} = \{\lambda_1, \cdots, \lambda_{n-r}\}$
- the final solution for $Ax = \mathbf{0}$ is $$\sum_{i = 1}^{n-r}a_i\lambda_i$$

Note that $r$ means the **rank** of $A$, which equals the number of the pivot columns of the reduced echelon form. Therefore, the solution for $Ax = \mathbf{b}$ is an **affine space** of $\mathbb{R}^n$ whose dimension is $n-r$.

Also, if we regard $A$ as a transformation matrix to convert the basis $\psi$ which spans $W\in \mathbb{R}^m$ to the basis $\phi$ which spans $V\in \mathbb{R}^n$, (**the dimension of the two vector space is equal to $k$**) $$f: V\rightarrow W\qquad\psi = A\phi.$$ Then the **image** of $f$ is the vector subspace that is spanned by the **columns of $A$**, i.e. $$r = \mathsf{dim}(\mathsf{img}(f))$$

The most important thing to perceive vector is that **it indicates the coordinates of the point with regard to the basis which it lies in.** As a result, mapping $V$ to $W$, although $n$ may less than $m$, the dimension of the vector space $W$ is equal to that of $V$. Consiquently, it makes no change when we want to describe a point in the new vector space $W$, because we still have to use $k$ coordinates to indicate it.

Natually, **the dimension of a vector space** is always less than or equal to **the elments number of the vector in it**. i.e.
$$k\le \mathsf{min}(n,m)$$
