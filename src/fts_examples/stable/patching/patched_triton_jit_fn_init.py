from fts_examples.stable.patching._patch_utils import _prepare_module_ctx
import re
from triton.runtime import jit  # noqa: F401


globals().update(_prepare_module_ctx('triton.runtime.jit', globals()))

# we ignore these for the entire file since we're using our global namespace trickeration to patch
# ruff: noqa: F821
# pyright: reportUndefinedVariable=false


def _new_init(self, fn, version=None, do_not_specialize=None, debug=None, noinline=None, repr=None,
              launch_metadata=None):
    do_not_specialize = do_not_specialize if do_not_specialize else []

    self.fn = fn
    self.module = fn.__module__
    self.version = version
    self.signature = inspect.signature(fn)
    self.do_not_specialize = do_not_specialize
    self.starting_line_number = inspect.getsourcelines(fn)[1]
    self.repr = lambda _: fn.__name__ if repr is None else repr(_)
    self.launch_metadata = launch_metadata

    self.params = []
    for i, param in enumerate(self.signature.parameters.values()):
        dns = do_not_specialize and (i in do_not_specialize or param.name in do_not_specialize)
        self.params.append(KernelParam(i, param, dns))

    # function source code (without decorators)
    self.src = textwrap.dedent(inspect.getsource(fn))
    #self.src = self.src[self.src.find("def"):]
    self.src = self.src[re.search(r"^def\s+\w+\s*\(", self.src, re.MULTILINE).start():]
    # cache of just-in-time compiled kernels
    self.cache = defaultdict(dict)
    self.hash = None
    # JITFunction can be instantiated as kernel
    # when called with a grid using __getitem__
    self.kernel = None
    self.debug = True if os.environ.get("TRITON_DEBUG", "0") == "1" else debug
    self.noinline = noinline

    # TODO(jlebar): Remove uses of these fields outside this file, then
    # remove the fields here.
    self.arg_names = [p.name for p in self.params]
    self.constexprs = [p.num for p in self.params if p.is_constexpr]

    # Hooks that will be called prior to executing "run"
    self.pre_run_hooks = []

    # reuse docs of wrapped function
    self.__doc__ = fn.__doc__
    self.__name__ = fn.__name__
    self.__globals__ = fn.__globals__
    self.__module__ = fn.__module__

# JITFunction.__init__ = _new_init
