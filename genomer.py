import contextlib
import re

def staticvar(**kwargs):
    def decorate(func):
        for k, v in kwargs.items():
            setattr(func, k, v)
        return func
    return decorate

class Program:
    def __init__(self, tab='    '):
        self.depth = 0
        self.tab = tab
        self.code = []

    def aline(self, s, *args, **kwargs):
        self.code.append(self.depth * self.tab + s.format(*args, **kwargs))
        return self

    def alines(self, s):
        self.code.extend((self.depth * self.tab + l for l in s.splitlines()))
        return self

    def get_code(self):
        return '\n'.join(self.code)

    def print_code(self):
        print(self.get_code())
        return self

    def eval_code(self):
        return eval(self.get_code())

    def exec_code(self):
        return exec(self.get_code())

    def indent(self, n=1):
        self.depth += n
        return self

    def dedent(self, n=1):
        self.depth -= n
        if self.depth < 0:
            raise SyntaxError('Internal over-dedenting')

        return self

    @contextlib.contextmanager
    def indentc(self, head=None, *args):
        if head is not None:
            self.aline(head, *args)
        self.indent()
        yield
        self.dedent()

    @contextlib.contextmanager
    def ifc(self, cond):
        with self.indentc('if ({}):'.format(cond)):
            yield

    def withifc(self, cond, content):
        with self.ifc(cond):
            self.alines(content)
        return self


class Parser:
    def __init__(self, debug=False):
        self.debug = debug
        self.whitespace = r'\s*' # Whitespace pattern must always match

        self.statuses = dict()

        self.program = Program()
        self.states = dict()
        self.max_state = -1

        self.registers = dict()
        self.max_register = -1

        self.rgxs = list()

        self.state_code = dict()

        self.root_node = None

        self.declare_register('STATUS')
        self.declare_register('STATE_D')
        self.declare_register('NEXT')
        self.declare_register('CUR_OFS')
        self.declare_register('DATA')
        self.declare_register('WHITESPACE')
        self.declare_status('FAIL')
        self.declare_status('SUCCESS')
        self.declare_status('MATCH')
        self.declare_status('NOMATCH')

        self.main_func = Program()
        self.main_func.aline('global #[*]')
        self.main_func.aline('#[@STATE_D] = #[^]')
        self.main_func.aline('#[@NEXT] = []')
        self.main_func.aline('#[@DATA] = []')
        self.main_func.aline('#[@CUR_OFS] = 0')
        self.main_func.aline('while #[@STATUS] != #[$FAIL] and #[@STATUS] != #[$SUCCESS]:')
        self.main_func.indent()

    def define_whitespace(self, pat):
        self.whitespace = pat

    def declare_status(self, name):
        n = len(self.statuses.keys())
        self.statuses[name] = n

        return self

    def declare_state(self, name):
        n = self.max_state + 1
        self.max_state += 1

        self.states[name] = n
        return self

    def define_state(self, name, code):
        assert(type(code) is Program)
        self.state_code[name] = code

    def declare_register(self, name):
        n = self.max_register + 1
        self.max_register += 1

        self.registers[name] = 'REG{}'.format(n)
        return self

    def declare_rgx(self, pat):
        self.rgxs.append(pat)
        return len(self.rgxs)-1

    def generate(self):
        self.program.aline('import re')

        # === header ===
        # --- declare states ---
        self.program.aline('# === states ===')
        for k, v in self.states.items():
            self.program.aline('# {}: {}'.format(k, v))

        self.program.aline('STATES = {}'.format(repr(list(self.states.values()))))


        # --- declare registers ---
        self.program.aline('# === registers ===')
        for k, v in self.registers.items():
            self.program.aline('# {}: {}'.format(k, v))
            self.program.aline('{} = None'.format(v))

        # --- declare grxs ---
        self.program.aline('# === patterns ===')
        self.program.aline('RGXS = [{}]'.format(', '.join(('re.compile(\'%s\')' % p for p in self.rgxs))))
        self.program.aline('#[@WHITESPACE] = re.compile(\'%s\')' % self.whitespace)

        # --- main func ---
        isf = True
        for n, c in self.state_code.items():
            self.main_func.aline('')
            self.main_func.aline('# if state is `%s`' % n)
            with self.main_func.indentc('%sif #[@STATE_D] == #[&%s]:'
                                        % ('' if isf else 'el', n)):
                self.main_func.alines(c.get_code())
            isf = False
        with self.program.indentc('def parse(s):'):
            self.program.alines(self.main_func.get_code())
            self.program.aline('return #[@DATA].pop()')




        # --- debugging ---
        with self.program.indentc('if __name__ == "__main__":'):
            self.program.aline('print(repr(parse("(testing)")))')

        # --- expand macros ---
        assert self.root_node is not None, 'No root node specified!'

        expanded = self.program.get_code()
        expanded = re.sub(r'#\[@([^\]]*)\]',
                          lambda m: '{}'.format(self.registers[m.group(1)]),
                          expanded)
        expanded = re.sub(r'#\[&([^\]]*)\]',
                          lambda m: '{}'.format(self.states[m.group(1)]),
                          expanded)
        expanded = re.sub(r'#\[\$([^\]]*)\]',
                          lambda m: '{}'.format(self.statuses[m.group(1)]),
                          expanded)
        expanded = re.sub(r'#\[\*\]', ', '.join(self.registers.values()),
                          expanded)
        expanded = re.sub(r'#\[\^\]', '{}'.format(self.states[self.root_node]),
                          expanded)

        #expanded = re.sub(r'#\[\^([^\]]*)\]',
        #                  lambda m: '{}'.format(self.rgxs[m.group(1)]),
        #                  expanded)

        return Program().alines(expanded)


    # this could be a context...
    def many0(self, state, _static={'id': 0}):
        name_s = 'many0-setup-'  + str(_static['id'])
        name_b = 'many0-before-' + str(_static['id'])
        name_a = 'many0-after-'  + str(_static['id'])
        _static['id'] += 1

        code = Program()

        code.aline('#[@STATE_D] = #[&%s]' % name_b)
        code.aline('#[@DATA].append([])')

        self.declare_state(name_s)
        self.define_state(name_s, code)

        code = Program()

        if self.debug:
            code.aline('print("{}: {}")', name_b, state)

        code.aline('#[@NEXT].append(#[&{}])', name_a)
        code.aline('#[@STATE_D] = #[&%s]' % state)
        code.aline('continue')

        self.declare_state(name_b)
        self.define_state(name_b, code)

        code = Program()
        if self.debug:
            code.aline('print("{}: {}")', name_a, state)

        with code.indentc('if #[@STATUS] == #[$MATCH]:'):
            code.aline('#[@STATE_D] = #[&{}]', name_b)
            code.aline('#[@DATA][-2].append(#[@DATA].pop())')
        with code.indentc('else:'):
            code.aline('#[@STATUS] = #[$MATCH]')
            code.aline('#[@STATE_D] = #[@NEXT].pop()')
            # should we pop the empty data here?
        code.aline('continue')

        self.declare_state(name_a)
        self.define_state(name_a, code)

        return name_s

    def rgx_ws(self, patid):
        return self.rgx(patid, consume_ws=True)

    def rgx(self, patid, consume_ws=False, _static={'id': 0}):
        name = 'rgx-'+str(_static['id'])
        _static['id'] += 1

        code = Program()
        if self.debug:
            code.aline('print("{0}: ({1}, re: %s, ofs: %s)" % (RGXS[{1}].pattern, #[@CUR_OFS]))', name, patid)

        if consume_ws:
            code.aline('# consume whitespace')
            code.aline('ws_start, ws_end = #[@WHITESPACE].match(s, #[@CUR_OFS]).span()')
        code.aline('# match rgx pattern')
        code.aline('m = RGXS[{}].match(s, {} #[@CUR_OFS])',
                    patid,
                    '(ws_end - ws_start) +' if consume_ws else '')

        with code.indentc('if m is None:'):
            code.aline('#[@STATUS] = #[$NOMATCH]')
            if self.debug:
                code.aline('print("NOMATCH")')
        with code.indentc('else:'):
            code.aline('#[@DATA].append(m)')
            code.aline('#[@STATUS] = #[$MATCH]')
            code.aline('#[@CUR_OFS] += {} (m.end() - m.start())', '(ws_end - ws_start) +' if consume_ws else '')
            if self.debug:
                code.aline('print("MATCH")')

        code.aline('#[@STATE_D] = #[@NEXT].pop()')
        code.aline('continue')

        self.declare_state(name)
        self.define_state(name, code)

        return name

    def do_parse(self, *states, root_node=False, _static={'id': 0}):
        states = list(states)
        name_b = 'do-parse-before-'  + str(_static['id'])
        _name_a = 'do-parse-after-'  + str(_static['id'])
        name_last = 'do-parse-exit-' + str(_static['id'])
        _static['id'] += 1

        if root_node:
            self.root_node = name_b

        code = Program()
        if self.debug:
            code.aline('print("{}: {}")', name_b, states[0])

        code.aline(
            '#[@NEXT].append(#[&{}])',
            name_last if len(states) == 1 else (_name_a + '-0'))

        code.aline('#[@STATE_D] = #[&%s]' % states[0])
        code.aline('#[@DATA].append([])')
        code.aline('continue')

        self.declare_state(name_b)
        self.define_state(name_b, code)

        max = len(states[1:])-1
        for i, state in enumerate(states[1:]):
            name_a      = _name_a + '-' + str(i)
            name_a_next = (_name_a + '-' + str(i+1)) if i != max else None

            code = Program()
            if self.debug:
                code.aline('print("{}: {} -> {}")', name_a, state, name_a_next)

            # If matched: try next
            with code.indentc('if #[@STATUS] == #[$MATCH]:'):
                if self.debug:
                    code.aline('print("trying: {}")', state)
                code.aline('#[@NEXT].append(#[&{}])', name_last if i == max else name_a_next)
                code.aline('#[@STATE_D] = #[&%s]' % state)
                code.aline('#[@DATA][-2].append(#[@DATA].pop())')

            # If didnt match: status=FAIL    <- root node
            #                 status=NOMATCH <- normal node
            with code.indentc('else:'):
                if root_node:
                    code.aline('print("MISMATCH IN ROOT DOPARSE")')
                    code.aline('#[@STATUS] = #[$FAIL]')
                else:
                    code.aline('#[@DATA].pop()')
                    code.aline('#[@STATUS] = #[$NOMATCH]')
                    code.aline('#[@STATE_D] = #[@NEXT].pop()')
            code.aline('continue')

            print('DECLARING:', name_a)
            self.declare_state(name_a)
            self.define_state(name_a, code)

        code = Program()
        with code.indentc('if #[@STATUS] == #[$MATCH]:'):
            code.aline('#[@DATA][-2].append(#[@DATA].pop())')
            if root_node:
                code.aline('print("success!, full match!")')
                code.aline('#[@STATUS] = #[$SUCCESS]')
                #code.aline('break') # make status SUCCESS instead to break
            else:
                code.aline('#[@STATE_D] = #[@NEXT].pop()')
        with code.indentc('else:'):
            if root_node:
                code.aline('print("MISMATCH IN ROOT DOPARSE")')
                code.aline('#[@STATUS] = #[$FAIL]')
            else:
                code.aline('#[@DATA].pop()')
                code.aline('#[@STATUS] = #[$NOMATCH]')
                code.aline('#[@STATE_D] = #[@NEXT].pop()')
        code.aline('continue')

        self.declare_state(name_last)
        self.define_state(name_last, code)

        return name_b

if __name__ == '__main__':
    par = Parser(debug=False)

    LP = par.declare_rgx(r'\(')
    RP = par.declare_rgx(r'\)')
    name = par.declare_rgx(r'[a-zA-Z_][a-zA-Z\-_0-9!?]*')

    ident = par.rgx_ws(name)

    expr = par.do_parse(par.rgx_ws(LP),
                        ident,
                        par.rgx_ws(RP))

    par.do_parse(
        par.many0(expr),
        root_node=True)

    code = par.generate().print_code().get_code()

    with open('genny.py', 'w') as f:
        print(code, file=f)
