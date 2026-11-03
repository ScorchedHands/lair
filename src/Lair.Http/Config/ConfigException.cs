namespace Lair.Host.Config;

/// <summary>
/// Raised when a configuration file cannot be read, is malformed, or fails a
/// validation rule. Every message names the offending key and the accepted values.
/// </summary>
public sealed class ConfigException : Exception
{
    public ConfigException(string message)
        : base(message)
    {
    }

    public ConfigException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}
