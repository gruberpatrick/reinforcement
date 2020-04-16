
class Memory:

    _m = []

    def __init__(self, fields=[]):

        self._fields = set(fields)

    def reset(self):
        """
        Reset the memory object.
        """

        self._m = []

    def add(self, **kwargs):
        """
        Add a row to memory. Checks the keys that are passed in.
        :param **kwargs: Keyword arguments are parsed and checked against the keywords passed in during
                         initialization.
        :return: Bool, True if successful (checks passed), False otherwise.
        """

        value = {}
        keys = set()

        for key in kwargs:

            # if the key is not in the defined keys, exit execution;
            if key not in self._fields:
                return False

            # keep track of keys and save the values;
            keys.add(key)
            value[key] = kwargs[key]

        # not all values provided, exit execution;
        if self._fields != keys:
            return False

        self._m.append(value)
        return True
