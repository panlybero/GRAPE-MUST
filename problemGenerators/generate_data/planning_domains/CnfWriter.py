import logging
import itertools
import io


class CnfWriter:
    '''
    Class for converting plan formulas to CNF
    Slighly modified from https://github.com/aibasel/pyperplan to make IO more flexible

    '''

    def __init__(self):
        self.cnf_file = io.StringIO()
        self.vars_to_numbers = {}
        self.is_written = False

    def _print_clause(self, clause):
        print(
            " ".join(str(self._literal_to_int(literal))
                     for literal in clause) + " 0",
            file=self.cnf_file,
        )

    def _print_clauses(self, clauses):
        for clause in clauses:
            self._print_clause(clause)

    def _get_aux_var(self):
        return next(self.count)

    def _literal_to_int(self, literal):
        if type(literal) is int:
            return literal
        negated = literal.startswith("not-")
        if negated:
            # remove the 'not-' string
            literal = literal[4:]
        if literal in self.vars_to_numbers:
            number = self.vars_to_numbers[literal]
        else:
            number = next(self.count)
            self.vars_to_numbers[literal] = number
        if negated:
            number = -number
        return number

    def _get_aux_clauses_for_iff(self, iff):
        a2, a1 = iff.split("<->")
        return [
            [iff, a2, a1],
            [iff, "not-" + a2, "not-" + a1],
            ["not-" + iff, a2, "not-" + a1],
            ["not-" + iff, "not-" + a2, a1],
        ]

    def _get_aux_clauses_for_and(self, var1, var2):
        # aux = '{0}AND{1}'.format(var1, var2)
        aux = self._get_aux_var()
        not_var1 = "not-" + var1 if type(var1) is str else -var1
        not_var2 = "not-" + var2 if type(var2) is str else -var2
        return aux, [[-aux, var1], [-aux, var2], [not_var1, not_var2, aux]]

    def write(self, formula):
        """Adds helper variables for all occurences of "a2<->a1" """

        self.count = itertools.count(start=1)
        self.vars_to_numbers = dict()

        aux_iff_vars = set()

        logging.debug("Writing minisat input file")
        # We omit specifying the number of vars and clauses because we don't
        # know those when we start writing the file

        while formula:
            disj = formula.pop(0)
            if not isinstance(disj, list):
                self._print_clause([disj])
                continue
            new_clause = []
            for conj in disj:
                if not isinstance(conj, list):
                    new_clause.append(conj)
                    continue
                # Add auxiliary vars for iffs
                for literal in conj:
                    if "<->" in literal and literal not in aux_iff_vars:
                        self._print_clauses(
                            self._get_aux_clauses_for_iff(literal))
                        aux_iff_vars.add(literal)
                # Turn list into one literal and add auxiliary clauses
                while len(conj) > 1:
                    var1 = conj.pop(0)
                    var2 = conj.pop(0)
                    aux_var, clauses = self._get_aux_clauses_for_and(
                        var1, var2)
                    conj.insert(0, aux_var)
                    self._print_clauses(clauses)
                assert len(conj) == 1, conj
                new_clause.append(conj[0])
            self._print_clause(new_clause)

        for key in list(self.vars_to_numbers):
            if "<->" in key:
                del self.vars_to_numbers[key]

        self.is_written = True

        return self.vars_to_numbers

    def get_cnf_str(self):
        assert self.is_written, "CNF not written yet"
        return self.cnf_file.getvalue()

    def get_cnf_ptr(self):
        assert self.is_written, "CNF not written yet"
        return self.cnf_file
