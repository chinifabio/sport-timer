use std::{ffi::CString, marker::PhantomData, sync::Arc};

use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::{ffi::c_str, types::PyTuple};
use renoir::block::structure::{BlockStructure, OperatorStructure};
use renoir::{Stream, operator::Operator};

#[derive(Debug)]
struct PythonWrapper {
    module: Py<PyModule>,
}

#[derive(Debug, Clone)]
struct PythonOperator<Op, I, O>
where
    I: for<'py> IntoPyObject<'py, Target = PyTuple> + Clone + Send,
    O: for<'py> FromPyObject<'py> + Clone + Send,
    Op: Operator<Out = I>,
{
    prev: Op,
    python: Arc<PythonWrapper>,
    _i: PhantomData<I>,
    _o: PhantomData<O>,
}

impl<Op, I, O> std::fmt::Display for PythonOperator<Op, I, O>
where
    I: for<'py> IntoPyObject<'py, Target = PyTuple> + Clone + Send,
    O: for<'py> FromPyObject<'py> + Clone + Send,
    Op: Operator<Out = I>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} -> Python", self.prev)
    }
}

impl<Op, I, O> Operator for PythonOperator<Op, I, O>
where
    I: for<'py> IntoPyObject<'py, Target = PyTuple> + Clone + Send,
    O: for<'py> FromPyObject<'py> + Clone + Send,
    Op: Operator<Out = I>,
{
    type Out = O;

    fn setup(&mut self, metadata: &mut renoir::ExecutionMetadata) {
        self.prev.setup(metadata);
    }

    fn next(&mut self) -> renoir::operator::StreamElement<Self::Out> {
        self.prev.next().map(|item| {
            Python::with_gil(|py| -> PyResult<O> {
                let next = self.python.module.getattr(py, "next")?;
                let result = next.call1(py, item)?;
                result.extract(py)
            })
            .expect("Failed to convert python result")
        })
    }

    fn structure(&self) -> BlockStructure {
        self.prev
            .structure()
            .add_operator(OperatorStructure::new::<O, _>("PythonWrapper"))
    }
}

pub trait PythonExt {
    fn python<O: for<'py> FromPyObject<'py> + Clone + Send>(
        self,
        code: &'static str,
    ) -> Stream<impl Operator<Out = O>>;
}

impl<Op, I> PythonExt for Stream<Op>
where
    I: for<'py> IntoPyObject<'py, Target = PyTuple> + Clone + Send,
    Op: Operator<Out = I>,
{
    fn python<O: for<'py> FromPyObject<'py> + Clone + Send>(
        self,
        code: &str,
    ) -> Stream<impl Operator<Out = O>> {
        let code = CString::new(code).expect("Failed to setup python code for python engine");
        let module = Python::with_gil(|py| {
            PyModule::from_code(
                py,
                &code,
                c_str!("renoir_model_runner.py"),
                c_str!("renoir_model_runner"),
            )
            .expect("Failed to create the python module")
            .into()
        });
        self.add_operator(|prev| PythonOperator {
            prev,
            python: Arc::new(PythonWrapper { module }),
            _i: PhantomData::<I>,
            _o: PhantomData::<O>,
        })
    }
}
