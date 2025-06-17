use std::{
    ffi::CString,
    marker::PhantomData,
    sync::{Arc, RwLock},
};

use pyo3::types::PyModule;
use pyo3::{ffi::c_str, types::PyTuple};
use pyo3::{prelude::*, types::PyDict};
use renoir::block::structure::{BlockStructure, OperatorStructure};
use renoir::{Stream, operator::Operator};

#[derive(Debug)]
struct PythonWrapper {
    code: String,
    module: Option<Py<PyModule>>,
}

#[derive(Debug, Clone)]
struct PythonOperator<Op, I, O>
where
    I: for<'py> IntoPyObject<'py, Target = PyTuple> + Clone + Send,
    O: for<'py> FromPyObject<'py> + Clone + Send,
    Op: Operator<Out = I>,
{
    prev: Op,
    py_handle: Arc<RwLock<PythonWrapper>>,
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
        let mut handle = self
            .py_handle
            .write()
            .expect("Failed to acquire write lock on PythonWrapper");
        if handle.module.is_none() {
            let code_c_str = CString::new(handle.code.as_str())
                .expect("Failed to create CString from Python code");
            let module = Python::with_gil(|py| {
                let locals = PyDict::new(py);
                py.run(
                    c_str!(r#"import sys; import os; envdata = f"Python version: {sys.version}\nPython executable: {sys.executable}\nPython path: {sys.path}\nPython environment variables: {os.environ}""#),
                    None,
                    Some(&locals),
                )
                .expect("Failed to run Python code for environment info");
                let envdata: String = locals.get_item("envdata").unwrap().unwrap().unbind().to_string();
                println!("{envdata}");
                PyModule::from_code(
                    py,
                    &code_c_str,
                    c_str!("renoir_model_runner.py"),
                    c_str!("renoir_model_runner"),
                )
                .expect("Failed to create the python module")
                .into()
            });
            handle.module = Some(module);
        }
    }

    fn next(&mut self) -> renoir::operator::StreamElement<Self::Out> {
        let handle = self
            .py_handle
            .read()
            .expect("Failed to acquire read lock on PythonWrapper");
        // Module should be initialized by setup, so we can unwrap
        let py_module = handle
            .module
            .as_ref()
            .expect("Python module not initialized. Call setup first.");

        self.prev.next().map(|item| {
            Python::with_gil(|py| -> PyResult<O> {
                let next_fn = py_module.getattr(py, "next")?;
                let result = next_fn.call1(py, item)?;
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
        code: &str,
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
        let wrapper = PythonWrapper {
            code: code.to_string(),
            module: None,
        };
        self.add_operator(|prev| PythonOperator {
            prev,
            py_handle: Arc::new(RwLock::new(wrapper)),
            _i: PhantomData::<I>,
            _o: PhantomData::<O>,
        })
    }
}
