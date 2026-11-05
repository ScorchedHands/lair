using System.Collections;

namespace Lair.Templating.Minja;

/// <summary>
/// Insertion-ordered string-keyed map used for minja maps and later namespace objects. Re-setting an existing
/// key updates its value but keeps its original position, matching Python dict behavior.
/// </summary>
public sealed class MinjaMap : IEnumerable<KeyValuePair<string, MinjaValue>>
{
    private readonly Dictionary<string, MinjaValue> _values = [];
    private readonly List<string> _order = [];

    public MinjaMap()
    {
    }

    public MinjaMap(IEnumerable<KeyValuePair<string, MinjaValue>> entries)
    {
        foreach (KeyValuePair<string, MinjaValue> entry in entries)
            Set(entry.Key, entry.Value);
    }

    public int Count => _order.Count;

    /// <summary>Keys in insertion order.</summary>
    public IReadOnlyList<string> Keys => _order;

    public bool ContainsKey(string key) => _values.ContainsKey(key);

    public bool TryGet(string key, out MinjaValue value)
    {
        if (_values.TryGetValue(key, out MinjaValue? found))
        {
            value = found;
            return true;
        }

        value = MinjaValue.None;
        return false;
    }

    /// <summary>Gets the value for a key, or none when the key is absent (undefined equivalence).</summary>
    public MinjaValue Get(string key) => TryGet(key, out MinjaValue value) ? value : MinjaValue.None;

    public void Set(string key, MinjaValue value)
    {
        if (!_values.ContainsKey(key))
            _order.Add(key);

        _values[key] = value;
    }

    public IEnumerator<KeyValuePair<string, MinjaValue>> GetEnumerator()
    {
        for (int i = 0; i < _order.Count; i++)
            yield return new KeyValuePair<string, MinjaValue>(_order[i], _values[_order[i]]);
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}

/// <summary>Immutable list payload for minja values.</summary>
public sealed class MinjaList : IReadOnlyList<MinjaValue>
{
    private readonly MinjaValue[] _items;

    public MinjaList(IEnumerable<MinjaValue> items) => _items = [.. items];

    public MinjaValue this[int index] => _items[index];

    public int Count => _items.Length;

    public IEnumerator<MinjaValue> GetEnumerator() => ((IEnumerable<MinjaValue>)_items).GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
