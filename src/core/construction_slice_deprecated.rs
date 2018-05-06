use core::{MatrixSliceMN, MatrixSliceMutMN, Scalar};
use core::dimension::{Dim, DimName, Dynamic, U1};
use core::matrix_slice::{SliceStorage, SliceStorageMut};

/*
 *
 * Slice constructors.
 *
 */
impl<'a, N: Scalar, R: Dim, C: Dim, RStride: Dim, CStride: Dim>
    MatrixSliceMN<'a, N, R, C, RStride, CStride>
{
    /// Creates, without bound-checking, a matrix slice from an array and with dimensions and strides specified by generic types instances.
    ///
    /// This method is unsafe because the input data array is not checked to contain enough elements.
    /// The generic types `R`, `C`, `RStride`, `CStride` can either be type-level integers or integers wrapped with `Dynamic::new()`.
    #[inline]
    #[deprecated(note = "This constructor is being removed to avoid the incoherent_fundamental_impls lint error. Use ::from_slice_with_strides_generic_unchecked instead.")]
    pub unsafe fn new_with_strides_generic_unchecked(
        data: &'a [N],
        start: usize,
        nrows: R,
        ncols: C,
        rstride: RStride,
        cstride: CStride,
    ) -> Self {
        let data = SliceStorage::from_raw_parts(
            data.as_ptr().offset(start as isize),
            (nrows, ncols),
            (rstride, cstride),
        );
        Self::from_data(data)
    }

    /// Creates a matrix slice from an array and with dimensions and strides specified by generic types instances.
    ///
    /// Panics if the input data array dose not contain enough elements.
    /// The generic types `R`, `C`, `RStride`, `CStride` can either be type-level integers or integers wrapped with `Dynamic::new()`.
    #[inline]
    #[deprecated(note = "This constructor is being removed to avoid the incoherent_fundamental_impls lint error. Use ::from_slice_with_strides_generic instead.")]
    pub fn new_with_strides_generic(
        data: &'a [N],
        nrows: R,
        ncols: C,
        rstride: RStride,
        cstride: CStride,
    ) -> Self {
        // NOTE: The assertion implements the following formula, but without subtractions to avoid
        // underflow panics:
        //      len >= (ncols - 1) * cstride + (nrows - 1) * rstride + 1
        assert!(
            data.len() + cstride.value() + rstride.value()
                >= ncols.value() * cstride.value() + nrows.value() * rstride.value() + 1,
            "Matrix slice: input data buffer to small."
        );

        unsafe { Self::from_slice_with_strides_generic_unchecked(data, 0, nrows, ncols, rstride, cstride) }
    }
}

impl<'a, N: Scalar, R: Dim, C: Dim, RStride: Dim, CStride: Dim>
    MatrixSliceMutMN<'a, N, R, C, RStride, CStride>
{
    /// Creates, without bound-checking, a mutable matrix slice from an array and with dimensions and strides specified by generic types instances.
    ///
    /// This method is unsafe because the input data array is not checked to contain enough elements.
    /// The generic types `R`, `C`, `RStride`, `CStride` can either be type-level integers or integers wrapped with `Dynamic::new()`.
    #[inline]
    #[deprecated(note = "This constructor is being removed to avoid the incoherent_fundamental_impls lint error. Use ::from_slice_with_strides_generic_unchecked instead.")]
    pub unsafe fn new_with_strides_generic_mut_unchecked(
        data: &'a mut [N],
        start: usize,
        nrows: R,
        ncols: C,
        rstride: RStride,
        cstride: CStride,
    ) -> Self {
        let data = SliceStorageMut::from_raw_parts(
            data.as_mut_ptr().offset(start as isize),
            (nrows, ncols),
            (rstride, cstride),
        );
        Self::from_data(data)
    }

    /// Creates a mutable matrix slice from an array and with dimensions and strides specified by generic types instances.
    ///
    /// Panics if the input data array dose not contain enough elements.
    /// The generic types `R`, `C`, `RStride`, `CStride` can either be type-level integers or integers wrapped with `Dynamic::new()`.
    #[inline]
    #[deprecated(note = "This constructor is being removed to avoid the incoherent_fundamental_impls lint error. Use ::from_slice_with_strides_generic instead.")]
    pub fn new_with_strides_generic_mut(
        data: &'a mut [N],
        nrows: R,
        ncols: C,
        rstride: RStride,
        cstride: CStride,
    ) -> Self {
        // NOTE: The assertion implements the following formula, but without subtractions to avoid
        // underflow panics:
        //      len >= (ncols - 1) * cstride + (nrows - 1) * rstride + 1
        assert!(
            data.len() + cstride.value() + rstride.value()
                >= ncols.value() * cstride.value() + nrows.value() * rstride.value() + 1,
            "Matrix slice: input data buffer to small."
        );

        unsafe {
            Self::from_slice_with_strides_generic_unchecked(data, 0, nrows, ncols, rstride, cstride)
        }
    }
}

impl<'a, N: Scalar, R: Dim, C: Dim> MatrixSliceMN<'a, N, R, C> {
    /// Creates, without bound-checking, a matrix slice from an array and with dimensions specified by generic types instances.
    ///
    /// This method is unsafe because the input data array is not checked to contain enough elements.
    /// The generic types `R` and `C` can either be type-level integers or integers wrapped with `Dynamic::new()`.
    #[inline]
    #[deprecated(note = "This constructor is being removed to avoid the incoherent_fundamental_impls lint error. Use ::from_slice_generic_unchecked instead.")]
    pub unsafe fn new_generic_unchecked(data: &'a [N], start: usize, nrows: R, ncols: C) -> Self {
        Self::from_slice_with_strides_generic_unchecked(data, start, nrows, ncols, U1, nrows)
    }

    /// Creates a matrix slice from an array and with dimensions and strides specified by generic types instances.
    ///
    /// Panics if the input data array dose not contain enough elements.
    /// The generic types `R` and `C` can either be type-level integers or integers wrapped with `Dynamic::new()`.
    #[inline]
    #[deprecated(note = "This constructor is being removed to avoid the incoherent_fundamental_impls lint error. Use ::from_slice_generic instead.")]
    pub fn new_generic(data: &'a [N], nrows: R, ncols: C) -> Self {
        Self::from_slice_with_strides_generic(data, nrows, ncols, U1, nrows)
    }
}

impl<'a, N: Scalar, R: Dim, C: Dim> MatrixSliceMutMN<'a, N, R, C> {
    /// Creates, without bound-checking, a mutable matrix slice from an array and with dimensions specified by generic types instances.
    ///
    /// This method is unsafe because the input data array is not checked to contain enough elements.
    /// The generic types `R` and `C` can either be type-level integers or integers wrapped with `Dynamic::new()`.
    #[inline]
    #[deprecated(note = "This constructor is being removed to avoid the incoherent_fundamental_impls lint error. Use ::from_slice_generic_unchecked instead.")]
    pub unsafe fn new_generic_mut_unchecked(
        data: &'a mut [N],
        start: usize,
        nrows: R,
        ncols: C,
    ) -> Self {
        Self::from_slice_with_strides_generic_unchecked(data, start, nrows, ncols, U1, nrows)
    }

    /// Creates a mutable matrix slice from an array and with dimensions and strides specified by generic types instances.
    ///
    /// Panics if the input data array dose not contain enough elements.
    /// The generic types `R` and `C` can either be type-level integers or integers wrapped with `Dynamic::new()`.
    #[inline]
    #[deprecated(note = "This constructor is being removed to avoid the incoherent_fundamental_impls lint error. Use ::from_slice_generic instead.")]
    pub fn new_generic_mut(data: &'a mut [N], nrows: R, ncols: C) -> Self {
        Self::from_slice_with_strides_generic(data, nrows, ncols, U1, nrows)
    }
}

macro_rules! impl_constructors(
    ($($Dims: ty),*; $(=> $DimIdent: ident: $DimBound: ident),*; $($gargs: expr),*; $($args: ident),*) => {
        impl<'a, N: Scalar, $($DimIdent: $DimBound),*> MatrixSliceMN<'a, N, $($Dims),*> {
            /// Creates a new matrix slice from the given data array.
            ///
            /// Panics if `data` does not contain enough elements.
            #[inline]
            #[deprecated(note = "This constructor is being removed to avoid the incoherent_fundamental_impls lint error. Use ::from_slice instead.")]
            pub fn new(data: &'a [N], $($args: usize),*) -> Self {
                Self::from_slice_generic(data, $($gargs),*)
            }

            /// Creates, without bound checking, a new matrix slice from the given data array.
            #[inline]
            #[deprecated(note = "This constructor is being removed to avoid the incoherent_fundamental_impls lint error. Use ::from_slice_unchecked instead.")]
            pub unsafe fn new_unchecked(data: &'a [N], start: usize, $($args: usize),*) -> Self {
                Self::from_slice_generic_unchecked(data, start, $($gargs),*)
            }
        }

        impl<'a, N: Scalar, $($DimIdent: $DimBound, )*> MatrixSliceMN<'a, N, $($Dims,)* Dynamic, Dynamic> {
            /// Creates a new matrix slice with the specified strides from the given data array.
            ///
            /// Panics if `data` does not contain enough elements.
            #[inline]
            #[deprecated(note = "This constructor is being removed to avoid the incoherent_fundamental_impls lint error. Use ::from_slice_with_strides instead.")]
            pub fn new_with_strides(data: &'a [N], $($args: usize,)* rstride: usize, cstride: usize) -> Self {
                Self::from_slice_with_strides_generic(data, $($gargs,)* Dynamic::new(rstride), Dynamic::new(cstride))
            }

            /// Creates, without bound checking, a new matrix slice with the specified strides from the given data array.
            #[inline]
            #[deprecated(note = "This constructor is being removed to avoid the incoherent_fundamental_impls lint error. Use ::from_slice_with_strides_unchecked instead.")]
            pub unsafe fn new_with_strides_unchecked(data: &'a [N], start: usize, $($args: usize,)* rstride: usize, cstride: usize) -> Self {
                Self::from_slice_with_strides_generic_unchecked(data, start, $($gargs,)* Dynamic::new(rstride), Dynamic::new(cstride))
            }
        }
    }
);

// FIXME: this is not very pretty. We could find a better call syntax.
impl_constructors!(R, C;                         // Arguments for Matrix<N, ..., S>
                   => R: DimName, => C: DimName; // Type parameters for impl<N, ..., S>
                   R::name(), C::name();         // Arguments for `_generic` constructors.
                   ); // Arguments for non-generic constructors.

impl_constructors!(R, Dynamic;
                   => R: DimName;
                   R::name(), Dynamic::new(ncols);
                   ncols);

impl_constructors!(Dynamic, C;
                   => C: DimName;
                   Dynamic::new(nrows), C::name();
                   nrows);

impl_constructors!(Dynamic, Dynamic;
                   ;
                   Dynamic::new(nrows), Dynamic::new(ncols);
                   nrows, ncols);

macro_rules! impl_constructors_mut(
    ($($Dims: ty),*; $(=> $DimIdent: ident: $DimBound: ident),*; $($gargs: expr),*; $($args: ident),*) => {
        impl<'a, N: Scalar, $($DimIdent: $DimBound),*> MatrixSliceMutMN<'a, N, $($Dims),*> {
            /// Creates a new mutable matrix slice from the given data array.
            ///
            /// Panics if `data` does not contain enough elements.
            #[inline]
            #[deprecated(note = "This constructor is being removed to avoid the incoherent_fundamental_impls lint error. Use ::from_slice instead.")]
            pub fn new(data: &'a mut [N], $($args: usize),*) -> Self {
                Self::from_slice_generic(data, $($gargs),*)
            }

            /// Creates, without bound checking, a new mutable matrix slice from the given data array.
            #[inline]
            #[deprecated(note = "This constructor is being removed to avoid the incoherent_fundamental_impls lint error. Use ::from_slice_unchecked instead.")]
            pub unsafe fn new_unchecked(data: &'a mut [N], start: usize, $($args: usize),*) -> Self {
                Self::from_slice_generic_unchecked(data, start, $($gargs),*)
            }
        }

        impl<'a, N: Scalar, $($DimIdent: $DimBound, )*> MatrixSliceMutMN<'a, N, $($Dims,)* Dynamic, Dynamic> {
            /// Creates a new mutable matrix slice with the specified strides from the given data array.
            ///
            /// Panics if `data` does not contain enough elements.
            #[inline]
            #[deprecated(note = "This constructor is being removed to avoid the incoherent_fundamental_impls lint error. Use ::from_slice_with_strides_mut instead.")]
            pub fn new_with_strides(data: &'a mut [N], $($args: usize,)* rstride: usize, cstride: usize) -> Self {
                Self::from_slice_with_strides_generic(
                    data, $($gargs,)* Dynamic::new(rstride), Dynamic::new(cstride))
            }

            /// Creates, without bound checking, a new mutable matrix slice with the specified strides from the given data array.
            #[inline]
            #[deprecated(note = "This constructor is being removed to avoid the incoherent_fundamental_impls lint error. Use ::from_slice_with_strides_unchecked instead.")]
            pub unsafe fn new_with_strides_unchecked(data: &'a mut [N], start: usize, $($args: usize,)* rstride: usize, cstride: usize) -> Self {
                Self::from_slice_with_strides_generic_unchecked(
                    data, start, $($gargs,)* Dynamic::new(rstride), Dynamic::new(cstride))
            }
        }
    }
);

// FIXME: this is not very pretty. We could find a better call syntax.
impl_constructors_mut!(R, C;                         // Arguments for Matrix<N, ..., S>
                       => R: DimName, => C: DimName; // Type parameters for impl<N, ..., S>
                       R::name(), C::name();         // Arguments for `_generic` constructors.
                       ); // Arguments for non-generic constructors.

impl_constructors_mut!(R, Dynamic;
                       => R: DimName;
                       R::name(), Dynamic::new(ncols);
                       ncols);

impl_constructors_mut!(Dynamic, C;
                       => C: DimName;
                       Dynamic::new(nrows), C::name();
                       nrows);

impl_constructors_mut!(Dynamic, Dynamic;
                       ;
                       Dynamic::new(nrows), Dynamic::new(ncols);
                       nrows, ncols);
